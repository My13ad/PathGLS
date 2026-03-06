import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import math
import os
import json
import hashlib
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from utils_img import MacenkoAugmentor
from bert_score import score as bert_score_func
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from config import Config
from llm_client import UniversalLLMClient
from utils_wsi import WSIProcessor

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[Warning] 'spacy' not found. Entity extraction will rely on simple splitting.")

PROMPT_GRAPH_EXTRACTION = """
As a Pathology Assistant, extract semantic relationships from the text into structured triplets.
Format: JSON list of objects [{"sub": "...", "rel": "...", "obj": "..."}].

Rules:
1. Extract core medical assertions.
2. Split complex sentences into atomic triplets.
3. Normalize entities (e.g., "no atypia" -> sub: "atypia", rel: "is", obj: "absent").
4. Ignore non-factual statements.

Example Input: "The nuclei are enlarged."
Example Output: [{"sub": "nuclei", "rel": "are", "obj": "enlarged"}]

Text to process: "{text}"
"""


class HighResCLIPEncoder:
    def __init__(self, clip_model, clip_processor, device, patch_size=224, stride=224):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

    def get_image_features(self, full_image):
        w, h = full_image.size
        if w < self.patch_size or h < self.patch_size:
            inputs = self.processor(images=full_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                return self.model.get_image_features(**inputs)

        patches = []
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                x1 = min(x + self.patch_size, w)
                y1 = min(y + self.patch_size, h)
                x0 = x1 - self.patch_size if x1 - self.patch_size >= 0 else 0
                y0 = y1 - self.patch_size if y1 - self.patch_size >= 0 else 0
                box = (x0, y0, x1, y1)
                patch = full_image.crop(box)
                patches.append(patch)

        batch_size = 32
        all_features = []
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i : i + batch_size]
            inputs = self.processor(
                images=batch_patches, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                all_features.append(feats)

        if not all_features:
            return torch.zeros(1, 512).to(self.device)

        patch_features = torch.cat(all_features, dim=0)
        global_feature = torch.max(patch_features, dim=0, keepdim=True)[0]
        return global_feature


class MedicalKnowledgeEngine:
    def __init__(self, nli_model, nli_tokenizer, device):
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_sci_md")
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = None
        self.encoder = nli_model.base_model if hasattr(nli_model, "base_model") else nli_model
        self.tokenizer = nli_tokenizer
        self.device = device

    def extract_entities(self, text):
        if not text:
            return []
        entities = set()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                clean_text = ent.text.lower().strip()
                if len(clean_text) > 2:
                    entities.add(clean_text)
        else:
            words = text.split()
            for w in words:
                if len(w) > 4:
                    entities.add(w.lower().strip(",."))
        return list(entities)

    def get_embedding(self, text_list):
        if not text_list:
            return None
        inputs = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def calculate_concept_overlap(self, visual_text, diagnosis_text):
        vis_ents = self.extract_entities(visual_text)
        diag_ents = self.extract_entities(diagnosis_text)
        if not diag_ents:
            return 0.5
        if not vis_ents:
            return 0.0
        vis_embs = self.get_embedding(vis_ents)
        diag_embs = self.get_embedding(diag_ents)
        if vis_embs is None or diag_embs is None:
            return 0.0
        sim_matrix = cosine_similarity(diag_embs, vis_embs)
        threshold = 0.85
        matched_count = 0
        for i in range(len(diag_ents)):
            max_sim = np.max(sim_matrix[i])
            if max_sim > threshold:
                matched_count += 1
        return matched_count / len(diag_ents)


class GraphLogicEngine:
    def __init__(self, llm_client, nli_model, nli_tokenizer, device, cache_ref):
        self.llm_client = llm_client
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.device = device
        self.cache = cache_ref
        self.entailment_idx = 0
        if hasattr(nli_model.config, "id2label"):
            for idx, label in nli_model.config.id2label.items():
                if "entailment" in label.lower():
                    self.entailment_idx = int(idx)
                    break

    def extract_graph(self, text):
        if not text or len(text) < 5:
            return []

        text_hash = hashlib.md5(("GRAPH_" + text).encode("utf-8")).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]

        prompt = PROMPT_GRAPH_EXTRACTION.format(text=text)

        try:
            res = self.llm_client.chat_complete(prompt, temperature=0.0, image_path=None)
            clean_res = res.strip()

            json_match = re.search(r"```json(.*?)```", clean_res, re.DOTALL)
            if json_match:
                clean_res = json_match.group(1).strip()
            else:
                code_match = re.search(r"```(.*?)```", clean_res, re.DOTALL)
                if code_match:
                    clean_res = code_match.group(1).strip()

            triplets_json = json.loads(clean_res)

            triplet_sentences = []
            if isinstance(triplets_json, list):
                for t in triplets_json:
                    if isinstance(t, dict) and "sub" in t and "rel" in t and "obj" in t:
                        sent = f"{t['sub']} {t['rel']} {t['obj']}"
                        triplet_sentences.append(sent)
        except Exception:
            triplet_sentences = []

        self.cache[text_hash] = triplet_sentences
        return triplet_sentences

    def calculate_graph_support(self, premise_text, hypothesis_text):
        prem_triplets = self.extract_graph(premise_text)
        hyp_triplets = self.extract_graph(hypothesis_text)
        if not hyp_triplets:
            return 0.5
        if not prem_triplets:
            return 0.0
        pairs = []
        for th in hyp_triplets:
            for tp in prem_triplets:
                pairs.append((tp, th))
        if not pairs:
            return 0.0
        batch_size = 16
        max_support_scores = [0.0] * len(hyp_triplets)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            try:
                inputs = self.nli_tokenizer(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                    probs = F.softmax(logits, dim=1)
                    scores = probs[:, self.entailment_idx].cpu().numpy()
                for k, score in enumerate(scores):
                    global_idx = i + k
                    hyp_idx = global_idx // len(prem_triplets)
                    if hyp_idx < len(max_support_scores):
                        if score > max_support_scores[hyp_idx]:
                            max_support_scores[hyp_idx] = score
            except Exception:
                continue
        return np.mean(max_support_scores)


class AutoCalibEvaluator:
    def __init__(self, llm_client=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f">>> [EvalEngine] Device: {self.device}")

        self.nli_model_name = "ChamW/roberta-base-mednli"
        print(f">>> [EvalEngine] Loading Logic Model: {self.nli_model_name}")
        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            ).to(self.device)
            self.nli_model.eval()
            self.entailment_idx = 0
            if hasattr(self.nli_model.config, "id2label"):
                for idx, label in self.nli_model.config.id2label.items():
                    if "entailment" in label.lower():
                        self.entailment_idx = int(idx)
                        break
        except Exception as e:
            print(f"[Error] Loading MedNLI: {e}. Fallback to DeBERTa.")
            self.nli_model_name = "cross-encoder/nli-deberta-v3-base"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, use_fast=False)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name
            ).to(self.device)
            self.entailment_idx = 1

        self.visual_model_name = "vinid/plip"
        print(f">>> [EvalEngine] Loading Visual Model: {self.visual_model_name}")
        try:
            self.clip_model = CLIPModel.from_pretrained(self.visual_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.visual_model_name)
        except Exception:
            print("[Warning] PLIP load failed. Fallback to Standard CLIP.")
            self.clip_model = CLIPModel.from_pretrained(Config.CLIP_MODEL).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)

        self.high_res_encoder = HighResCLIPEncoder(
            self.clip_model, self.clip_processor, self.device
        )
        if llm_client is not None:
            print(">>> [EvalEngine] Reusing existing LLM Client (Memory Saved).")
            self.llm_client = llm_client
        else:
            print(">>> [EvalEngine] Initializing new LLM Client...")
            self.llm_client = UniversalLLMClient()
        self.feature_cache = self._load_cache()

        self.med_engine = MedicalKnowledgeEngine(
            self.nli_model, self.nli_tokenizer, self.device
        )
        self.graph_engine = GraphLogicEngine(
            self.llm_client,
            self.nli_model,
            self.nli_tokenizer,
            self.device,
            self.feature_cache,
        )
        self.logic_graph_engine = self._init_logic_graph_engine()

        self.thresh_grounding = 0.20
        self.thresh_stability = 0.80
        self.sigmoid_k = 60.0
        self.base_weights = {"grounding": 0.4, "logic": 0.3, "stability": 0.3}

    def _init_logic_graph_engine(self):
        original_mode = Config.MODEL_SOURCE
        try:
            # Force graph extraction for logic-consistency to use remote API model from Config.
            Config.MODEL_SOURCE = "api"
            remote_client = UniversalLLMClient()
            return GraphLogicEngine(
                remote_client,
                self.nli_model,
                self.nli_tokenizer,
                self.device,
                self.feature_cache,
            )
        except Exception as e:
            print(f"[Warning] Remote graph engine init failed: {e}. Fallback to default graph engine.")
            return self.graph_engine
        finally:
            Config.MODEL_SOURCE = original_mode

    def _load_cache(self):
        if os.path.exists(Config.FILE_FEATURE_CACHE):
            try:
                with open(Config.FILE_FEATURE_CACHE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(Config.FILE_FEATURE_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.feature_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _get_text_hash(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest() if text else "empty"

    def parse_pathology_text(self, text):
        if not text:
            return {"visual": "", "diagnosis": "", "full": ""}
        split_pattern = r"(?:DIAGNOSIS|CONCLUSION|IMPRESSION|INTERPRETATION|DIAGNOSTIC OPINION|FINAL DIAGNOSIS)[\s:_]*"
        parts = re.split(split_pattern, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            diagnosis = parts[-1].strip()
            visual = " ".join(parts[:-1]).strip()
        else:
            lines = text.strip().split("\n")
            if len(lines) >= 2:
                diagnosis = lines[-1].strip()
                visual = "\n".join(lines[:-1]).strip()
            else:
                visual = text
                diagnosis = text
        return {"visual": visual, "diagnosis": diagnosis, "full": text}

    def _get_safe_image_path(self, raw_path_from_json):
        filename = os.path.basename(raw_path_from_json)
        full_path = os.path.join(Config.IMAGE_ROOT, filename)
        return full_path.replace("\\", "/")

    def extract_triplets(self, text):
        if not text or len(text) < 5:
            return []
        text_hash = self._get_text_hash(text)
        if text_hash in self.feature_cache:
            return self.feature_cache[text_hash]
        prompt = f"{Config.PROMPT_FEATURE_EXTRACTION}\n\nText: {text}"
        try:
            res = self.llm_client.chat_complete(prompt, temperature=0.0)
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0]
            elif "```" in res:
                res = res.split("```")[0]
            triplets = json.loads(res.strip())
            if not isinstance(triplets, list):
                triplets = ["nuclei are visible"]
        except Exception:
            triplets = ["nuclei are visible"]
        self.feature_cache[text_hash] = triplets
        self._save_cache()
        return triplets

    def calculate_grounding_final(self, img_path, text):
        if not os.path.exists(img_path):
            return 0.0
        triplets = self.extract_triplets(text)
        if not triplets:
            return 0.20
        try:
            full_image = Image.open(img_path).convert("RGB")
            image_features = self.high_res_encoder.get_image_features(full_image)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_inputs = self.clip_processor(
                text=triplets[:15], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            similarity = image_features @ text_features.t()
            score = similarity.mean().item()
            raw_score = max(0.0, score)
        except Exception:
            raw_score = 0.20
        return 1.0 / (1.0 + math.exp(-self.sigmoid_k * (raw_score - self.thresh_grounding)))

    def calculate_logic_consistency(self, text):
        if not text or len(text) < 10:
            return 1.0

        graph_engine = self.logic_graph_engine if self.logic_graph_engine is not None else self.graph_engine
        triplets = graph_engine.extract_graph(text)
        clean_triplets = []
        seen = set()
        for t in triplets:
            s = str(t).strip()
            if len(s) < 5 or s in seen:
                continue
            seen.add(s)
            clean_triplets.append(s)

        if len(clean_triplets) < 2:
            return 1.0

        check_list_a = clean_triplets[:6]
        check_list_b = clean_triplets[-6:]
        pairs_premise = []
        pairs_hypothesis = []
        for s1 in check_list_a:
            for s2 in check_list_b:
                if s1 == s2:
                    continue
                pairs_premise.append(s1)
                pairs_hypothesis.append(s2)
        if not pairs_premise:
            return 1.0
        try:
            inputs = self.nli_tokenizer(
                pairs_premise,
                pairs_hypothesis,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
            contradiction_idx = -1
            if hasattr(self.nli_model.config, "label2id"):
                l2i = self.nli_model.config.label2id
                for k in ["contradiction", "Contradiction", "CONTRADICTION"]:
                    if k in l2i:
                        contradiction_idx = l2i[k]
                        break
            if contradiction_idx == -1:
                if "mednli" in self.nli_model_name.lower():
                    contradiction_idx = 2
                else:
                    contradiction_idx = 0
            contradiction_probs = probs[:, contradiction_idx]
            # Top-k aggregation: focus on the most contradictory sentence pairs.
            topk_ratio = 0.3
            k = max(1, int(math.ceil(contradiction_probs.numel() * topk_ratio)))
            topk_vals, _ = torch.topk(contradiction_probs, k=k, largest=True)
            max_contradiction_prob = topk_vals.mean().item()
        except Exception:
            return 0.5
        return float(1.0 - max_contradiction_prob)

    def calculate_stability_final(self, t1, t2):
        if not t1 or not t2:
            return 0.0
        d1 = self.parse_pathology_text(t1)["diagnosis"]
        d2 = self.parse_pathology_text(t2)["diagnosis"]
        if len(d1) < 5:
            d1 = t1
        if len(d2) < 5:
            d2 = t2

        bert_val = 0.0
        try:
            _, _, f1 = bert_score_func(
                [d1],
                [d2],
                lang="en",
                model_type="roberta-large",
                device=self.device,
                verbose=False,
            )
            score = f1.item()
            bert_val = 1.0 if score > self.thresh_stability else max(0.0, score)
        except Exception:
            bert_val = 0.0

        graph_fwd = self.graph_engine.calculate_graph_support(premise_text=d1, hypothesis_text=d2)
        graph_bwd = self.graph_engine.calculate_graph_support(premise_text=d2, hypothesis_text=d1)
        graph_val = (graph_fwd + graph_bwd) / 2.0

        final_stability = 0.2 * bert_val + 0.8 * graph_val

        self._save_cache()
        return final_stability

    def compute_final_score(self, g, l, s):
        w_g = self.base_weights["grounding"]
        w_l = self.base_weights["logic"]
        w_s = self.base_weights["stability"]

        penalty = 1.0
        if l < 0.5:
            penalty *= 0.6
        if g < 0.25:
            penalty *= 0.8

        raw_score = (w_g * g) + (w_l * l) + (w_s * s)
        return raw_score * penalty

    def process_roi_dataset(self, sens_data, gen_data_map=None):
        results = []
        augmentor = MacenkoAugmentor()
        w_visual = 0.5
        w_semantic = 0.5

        os.makedirs(Config.TEMP_PATCH_DIR, exist_ok=True)

        for entry in tqdm(sens_data, desc="Processing ROI (Control/Visual/Attack)"):
            img_id = entry.get("image_id", "unknown")
            roi_path = entry.get("original_path") or entry.get("image_path")
            roi_full_path = self._get_safe_image_path(roi_path)
            if not os.path.exists(roi_full_path):
                print(f"[Error] ROI image not found: {roi_full_path}")
                continue

            t_ctrl = entry.get("text_control")
            if not t_ctrl:
                t_ctrl = self.llm_client.chat_complete(
                    prompt=Config.PROMPT_NEUTRAL,
                    image_path=roi_full_path,
                )

            t_visual = None
            if gen_data_map:
                cached = gen_data_map.get(img_id, {})
                t_visual = cached.get("response_augmented")

            if not t_visual:
                temp_out = os.path.join(Config.TEMP_PATCH_DIR, f"aug_{img_id}.png")
                success = augmentor.augment(roi_full_path, temp_out)
                if success:
                    t_visual = self.llm_client.chat_complete(
                        prompt=Config.PROMPT_NEUTRAL,
                        image_path=temp_out,
                    )
                else:
                    t_visual = t_ctrl

            t_attack = self.llm_client.chat_complete(
                prompt=Config.PROMPT_ATTACK_MALIGNANT,
                image_path=roi_full_path,
            )

            l_score = self.calculate_logic_consistency(t_ctrl)
            g_score = self.calculate_grounding_final(roi_full_path, t_ctrl)

            s_visual = self.calculate_stability_final(t_ctrl, t_visual)
            s_semantic = self.calculate_stability_final(t_ctrl, t_attack)
            s_final = (w_visual * s_visual) + (w_semantic * s_semantic)

            score_full = self.compute_final_score(g_score, l_score, s_final)

            results.append({
                "Image_ID": img_id,
                "Report_Control": t_ctrl,
                "Report_Visual": t_visual,
                "Report_Attack": t_attack,
                "Metric_Grounding": g_score,
                "Metric_Logic": l_score,
                "Metric_Stability_Visual": s_visual,
                "Metric_Stability_Semantic": s_semantic,
                "Metric_Stability_Weighted": s_final,
                "Score_Full": score_full,
            })

        return pd.DataFrame(results), None

    def process_wsi_dataset(self, sens_data, subject_client=None):
        results = []
        augmentor = MacenkoAugmentor()
        w_visual = 0.5
        w_semantic = 0.5
        os.makedirs(Config.TEMP_PATCH_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_AUG_DIR, exist_ok=True)
        gen_client = subject_client if subject_client is not None else self.llm_client

        for entry in tqdm(sens_data, desc="Processing WSI (Control/Visual/Attack)"):
            img_id = entry.get("image_id", "unknown")
            wsi_path = entry.get("original_path") or entry.get("image_path")
            wsi_full_path = self._get_safe_image_path(wsi_path)
            if not os.path.exists(wsi_full_path):
                print(f"[Error] WSI not found: {wsi_full_path}")
                continue

            try:
                processor = WSIProcessor(wsi_full_path, patch_size=Config.WSI_PATCH_SIZE)
                patches_data = processor.sample_patches(num_patches=Config.WSI_SAMPLE_NUM)
            except Exception as e:
                print(f"[Error] WSI Processing failed for {img_id}: {e}")
                continue

            if not patches_data:
                continue

            texts_ctrl = []
            texts_visual = []
            texts_attack = []
            patch_logic_scores = []
            patch_grounding_scores = []

            for p_item in patches_data:
                patch_img = p_item["patch_img"]
                patch_id = f"{img_id}_{len(texts_ctrl)}"
                temp_in = os.path.join(Config.TEMP_PATCH_DIR, f"{patch_id}.png")
                temp_out = os.path.join(Config.TEMP_AUG_DIR, f"aug_{patch_id}.png")
                patch_img.save(temp_in)

                t_ctrl = gen_client.inference(temp_in, prompt_text=Config.PROMPT_NEUTRAL)
                texts_ctrl.append(t_ctrl)

                l_c = self.calculate_logic_consistency(t_ctrl)
                patch_logic_scores.append(l_c)
                g_c = self.calculate_grounding_final(temp_in, t_ctrl)
                patch_grounding_scores.append(g_c)

                success = augmentor.augment(temp_in, temp_out)
                if success:
                    t_visual = gen_client.inference(temp_out, prompt_text=Config.PROMPT_NEUTRAL)
                else:
                    t_visual = t_ctrl
                texts_visual.append(t_visual)

                t_attack = gen_client.inference(temp_in, prompt_text=Config.PROMPT_ATTACK_MALIGNANT)
                texts_attack.append(t_attack)

            combo_ctrl = "\n\n".join([f"Region {i+1}: {t}" for i, t in enumerate(texts_ctrl)])
            combo_visual = "\n\n".join([f"Region {i+1}: {t}" for i, t in enumerate(texts_visual)])
            combo_attack = "\n\n".join([f"Region {i+1}: {t}" for i, t in enumerate(texts_attack)])

            report_ctrl = gen_client.generate_mil_summary(combo_ctrl)
            report_visual = gen_client.generate_mil_summary(combo_visual)
            report_attack = gen_client.generate_mil_summary(combo_attack)

            s_visual_wsi = self.calculate_stability_final(report_ctrl, report_visual)
            s_semantic_wsi = self.calculate_stability_final(report_ctrl, report_attack)
            s_final_wsi = (w_visual * s_visual_wsi) + (w_semantic * s_semantic_wsi)

            avg_logic = np.mean(patch_logic_scores) if patch_logic_scores else 0.0
            avg_grounding = np.mean(patch_grounding_scores) if patch_grounding_scores else 0.0

            score_full = self.compute_final_score(avg_grounding, avg_logic, s_final_wsi)

            results.append({
                "Image_ID": img_id,
                "Report_Control": report_ctrl,
                "Report_Visual": report_visual,
                "Report_Attack": report_attack,
                "Metric_Logic": avg_logic,
                "Metric_Stability_Visual": s_visual_wsi,
                "Metric_Stability_Semantic": s_semantic_wsi,
                "Metric_Stability_Weighted": s_final_wsi,
                "Score_Full": score_full,
            })

        return pd.DataFrame(results), None

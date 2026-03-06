import os
import json
import torch
import gc
from tqdm import tqdm
import pandas as pd

from config import Config
from llm_client import UniversalLLMClient
from eval_engine import AutoCalibEvaluator
from utils_img import MacenkoAugmentor


def setup_directories():
    os.makedirs(Config.TEMP_PATCH_DIR, exist_ok=True)
    os.makedirs(Config.TEMP_AUG_DIR, exist_ok=True)
    out_dir = os.path.dirname(Config.FILE_FINAL_REPORT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_dataset():
    print(f">>> Loading dataset from {Config.INPUT_DATA_FILE}...")
    try:
        with open(Config.INPUT_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "PubMed" in data and "val" in data["PubMed"]:
                data = data["PubMed"]["val"]
        final_list = []
        if isinstance(data, list):
            final_list = data
        elif isinstance(data, dict):
            if "PubMed" in data:
                final_list = data["PubMed"]["val"]
            else:
                found = False
                for k in ["images", "data", "samples", "items"]:
                    if k in data and isinstance(data[k], list):
                        final_list = data[k]
                        found = True
                        break
                if not found:
                    final_list = [data]
        return final_list[:500]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def scan_wsi_files(root_dir):
    wsi_paths = []
    extensions = Config.WSI_EXTENSIONS
    print(f">>> Scanning WSI files in: {root_dir} ...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                wsi_paths.append(os.path.join(root, file))
    print(f">>> Found {len(wsi_paths)} WSI files.")
    return wsi_paths


def run_roi():
    setup_directories()
    print("\n=== Step 1: Initializing LLM ===")
    try:
        llm_client = UniversalLLMClient()
    except Exception as e:
        print(f"LLM Init Failed: {e}")
        return

    augmentor = MacenkoAugmentor()
    data_list = load_dataset()
    if not data_list:
        return

    print(f"\n=== Step 2: Generation (Total: {len(data_list)}) ===")

    sens_data_for_eval = []
    gen_data_map = {}

    for idx, item in enumerate(tqdm(data_list, desc="Generating")):
        raw_id = item.get("image_id") or item.get("img") or item.get("id")
        if not raw_id:
            continue
        raw_id = str(raw_id)

        img_filename = raw_id
        if not any(img_filename.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
            img_filename += ".jpg"

        original_img_path = os.path.join(Config.IMAGE_ROOT, img_filename)
        img_id = os.path.splitext(os.path.basename(img_filename))[0]

        if not os.path.exists(original_img_path):
            continue

        response_control = llm_client.chat_complete(
            prompt=Config.PROMPT_NEUTRAL,
            image_path=original_img_path,
            temperature=0.2,
        )
        if not response_control:
            response_control = "No findings."

        aug_img_path = os.path.join(Config.TEMP_PATCH_DIR, f"aug_{img_filename}")
        if not os.path.exists(aug_img_path):
            try:
                success = augmentor.augment(original_img_path, aug_img_path)
                if not success:
                    aug_img_path = original_img_path
            except Exception:
                aug_img_path = original_img_path

        response_augmented = llm_client.chat_complete(
            prompt=Config.PROMPT_NEUTRAL,
            image_path=aug_img_path,
            temperature=0.2,
        )
        if not response_augmented or len(response_augmented) < 5:
            response_augmented = response_control

        record = {
            "image_id": img_id,
            "original_path": original_img_path,
            "text_control": response_control,
        }
        sens_data_for_eval.append(record)

        gen_data_map[img_id] = {
            "response_augmented": response_augmented,
        }

    with open(Config.FILE_GEN_RESULTS, "w", encoding="utf-8") as f:
        json.dump({"sens_data": sens_data_for_eval, "gen_map": gen_data_map}, f, indent=2, ensure_ascii=False)

    print(f">>> Generation finished. Saved to {Config.FILE_GEN_RESULTS}")

    print("\n=== Step 3: Calculation & Scoring ===")
    cleanup_gpu()

    try:
        evaluator = AutoCalibEvaluator(llm_client=llm_client)
    except TypeError:
        print("Warning: Re-initializing Evaluator (High Memory Usage)")
        evaluator = AutoCalibEvaluator()

    df_results, _ = evaluator.process_roi_dataset(sens_data_for_eval, gen_data_map)

    if df_results is not None and not df_results.empty:
        df_results.to_excel(Config.FILE_FINAL_REPORT, index=False)
        print(f"Report saved to: {Config.FILE_FINAL_REPORT}")
        print("\n" + "=" * 60)
        print("[Evaluation Results Overview]")
        print(df_results[["Metric_Grounding", "Metric_Logic", "Metric_Stability_Weighted", "Score_Full"]].mean())
        print("=" * 60 + "\n")
    else:
        print("Evaluation failed.")


def run_wsi():
    setup_directories()

    original_mode = Config.MODEL_SOURCE
    Config.MODEL_SOURCE = "local"
    subject_client = UniversalLLMClient()

    if Config.WSI_JUDGE_MODE == "api":
        Config.MODEL_SOURCE = "api"
        try:
            judge_client = UniversalLLMClient()
        except Exception as e:
            print(f"Judge Model init failed: {e}")
            Config.MODEL_SOURCE = original_mode
            return
    else:
        judge_client = subject_client

    Config.MODEL_SOURCE = original_mode

    evaluator = AutoCalibEvaluator(llm_client=judge_client)

    wsi_files = scan_wsi_files(Config.IMAGE_ROOT)
    if not wsi_files:
        print("No WSI files found.")
        return

    sens_data = []
    for p in wsi_files:
        slide_id = os.path.splitext(os.path.basename(p))[0]
        sens_data.append({"image_id": slide_id, "original_path": p})

    df_results, _ = evaluator.process_wsi_dataset(sens_data, subject_client=subject_client)

    if df_results is not None and not df_results.empty:
        df_results.to_excel(Config.FILE_WSI_REPORT, index=False)
        print(f"All done! Results saved to {Config.FILE_WSI_REPORT}")
    else:
        print("No results generated.")


def main():
    if Config.RUN_MODE == "wsi":
        run_wsi()
    else:
        run_roi()


if __name__ == "__main__":
    main()

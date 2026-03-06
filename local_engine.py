import os
import torch
import warnings
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class LocalVLMEngine:
    def __init__(self, model_path, model_name, engine_type):
        self.engine_type = engine_type
        if self.engine_type == "engine_b":
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
            disable_torch_init()
            self._llava = {
                "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
                "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
                "conv_templates": conv_templates,
                "SeparatorStyle": SeparatorStyle,
                "tokenizer_image_token": tokenizer_image_token,
                "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
            }
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    load_4bit=False,
                    load_8bit=False,
                    device="cuda",
                )
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()

    def inference(self, image_path, prompt_text):
        if self.engine_type == "engine_b":
            return self._inference_engine_b(image_path, prompt_text)
        return self._inference_engine_a(image_path, prompt_text)

    def _inference_engine_b(self, image_path, prompt_text):
        try:
            images = None
            if image_path and os.path.exists(image_path):
                qs = self._llava["DEFAULT_IMAGE_TOKEN"] + "\n" + prompt_text
                image = Image.open(image_path).convert("RGB")
                image_tensor = (
                    self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                    .half()
                    .cuda()
                )
                images = image_tensor
            else:
                qs = prompt_text
                images = None

            conv = self._llava["conv_templates"]["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            final_prompt = conv.get_prompt()

            input_ids = (
                self._llava["tokenizer_image_token"](
                    final_prompt,
                    self.tokenizer,
                    self._llava["IMAGE_TOKEN_INDEX"],
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .cuda()
            )

            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            stop_str = conv.sep if conv.sep_style != self._llava["SeparatorStyle"].TWO else conv.sep2
            stopping_criteria = self._llava["KeywordsStoppingCriteria"]([stop_str], self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            return outputs.strip()
        except Exception as e:
            print(f"[Local-LLM Error] {e}")
            return ""

    def _inference_engine_a(self, image_path, prompt_text):
        try:
            messages = []
            content = []
            images = None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                content.append({"type": "image"})
                images = [image]
            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text.strip()
        except Exception as e:
            print(f"[Local-LLM Error] {e}")
            return ""

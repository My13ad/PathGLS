import requests
import json
import time
import base64
import os
from config import Config
from local_engine import LocalVLMEngine


class UniversalLLMClient:
    def __init__(self):
        self.mode = Config.MODEL_SOURCE
        if self.mode == "local":
            print(">>> Mode: LOCAL GPU (Loading Model...)")
            self.local_engine = LocalVLMEngine(
                model_path=Config.LOCAL_MODEL_PATH,
                model_name=Config.LOCAL_MODEL_BASE,
                engine_type=Config.LOCAL_ENGINE,
            )
        else:
            print(">>> Mode: REMOTE API")
            self.api_key = Config.API_KEY
            self.base_url = Config.API_HOST.rstrip("/")
            self.model_name = Config.API_MODEL_NAME
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

    def chat_complete(self, prompt, system_prompt=None, image_path=None, temperature=0.0):
        if self.mode == "local":
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser Task:\n{prompt}"
            return self.local_engine.inference(image_path, full_prompt)
        return self._call_remote_api(prompt, system_prompt, image_path, temperature)

    def inference(self, image_path, prompt_text, temperature=0.0):
        return self.chat_complete(prompt_text, system_prompt=None, image_path=image_path, temperature=temperature)

    def _call_remote_api(self, prompt, system_prompt, image_path, temperature):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        for _ in range(3):
            try:
                url = f"{self.base_url}/chat/completions"
                res = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                    proxies={"http": None, "https": None},
                )
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"API Error: {e}")
                time.sleep(1)
        return "Error: API failed."

    def generate_mil_summary(self, combined_text):
        print(f">>> [MIL Aggregator] Sending massive context to {Config.API_MODEL_NAME}...")
        system_prompt = "You are a Senior Pathologist..."
        payload = {
            "model": Config.API_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {Config.API_KEY}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                res = requests.post(
                    f"{Config.API_HOST.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300,
                    proxies={"http": None, "https": None},
                )
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
                print(f"API Error ({res.status_code}): {res.text[:200]}")
            except requests.exceptions.ProxyError:
                print("Proxy Error: connection refused, retrying direct...")
            except Exception as e:
                print(f"Connection Error (Attempt {attempt + 1}): {e}")
            time.sleep(2)

        return "Error: API Request Failed (Check Network)."

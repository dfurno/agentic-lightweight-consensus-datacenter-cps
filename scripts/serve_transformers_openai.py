#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512


def build_prompt(messages: list[ChatMessage], tokenizer: Any) -> str:
    dict_messages = [{"role": message.role, "content": message.content} for message in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(dict_messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message.role}: {message.content}" for message in messages) + "\nassistant:"


def extract_assistant_text(decoded: str) -> str:
    text = decoded.strip()
    if "<|turn>model" in text:
        text = text.rsplit("<|turn>model", 1)[-1]
    text = re.sub(r"<\|channel\>thought\s*<channel\|>", "", text)
    text = text.replace("<turn|>", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("<pad>", "")
    return text.strip()


def normalize_diffusiongemma_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    system_parts = [message.content for message in messages if message.role == "system"]
    normalized: list[dict[str, str]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        normalized.append({"role": role, "content": message.content})
    if system_parts:
        system_text = "\n".join(system_parts)
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = f"{system_text}\n\n{normalized[0]['content']}"
        else:
            normalized.insert(0, {"role": "user", "content": system_text})
    return normalized


def first_real_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_app(model_id: str) -> FastAPI:
    app = FastAPI(title="Transformers OpenAI-compatible local server")
    is_diffusiongemma = "diffusiongemma" in model_id.lower()
    if is_diffusiongemma:
        try:
            from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        except ImportError as exc:
            raise RuntimeError(
                "DiffusionGemma requires a recent transformers release with "
                "DiffusionGemmaForBlockDiffusion. Run: pip install -U transformers accelerate"
            ) from exc
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if is_diffusiongemma and processor is not None:
            dict_messages = normalize_diffusiongemma_messages(request.messages)
            inputs = processor.apply_chat_template(
                dict_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(first_real_device(model))
        else:
            prompt = build_prompt(request.messages, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(first_real_device(model))
        with torch.no_grad():
            if is_diffusiongemma:
                output = model.generate(**inputs, max_new_tokens=request.max_tokens)
            else:
                output = model.generate(
                    **inputs,
                    do_sample=request.temperature > 0,
                    temperature=max(request.temperature, 1e-5),
                    max_new_tokens=request.max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
        if is_diffusiongemma:
            decoded = processor.decode(output[0], skip_special_tokens=False) if processor is not None else tokenizer.decode(output[0], skip_special_tokens=False)
            text = extract_assistant_text(decoded)
            completion_tokens = max(int(output[0].numel()) - int(inputs["input_ids"].numel()), 0)
        else:
            generated = output[0][inputs["input_ids"].shape[-1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            completion_tokens = int(generated.numel())
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].numel()),
                "completion_tokens": completion_tokens,
                "total_tokens": int(inputs["input_ids"].numel()) + completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

import argparse
import logging
import os
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
logger.info("LOCAL_LLM_MODEL = %s", MODEL_NAME)

app = FastAPI(title="Local LLM Server", version="2.3.0")

_tokenizer = None
_model = None


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_new_tokens: int = 160


def _load_model() -> None:
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    logger.info("Loading model '%s' — this may take a minute ...", MODEL_NAME)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    _model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Model '%s' loaded on %s.", MODEL_NAME, device)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "model_loaded": _model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }


@app.head("/generate")
def generate_head() -> Response:
    return Response(status_code=200)


@app.post("/generate")
async def generate(req: GenerateRequest) -> Dict[str, str]:
    _load_model()

    system_prompt = req.system_prompt or "You are a helpful assistant."
    prompt = req.prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=int(req.max_new_tokens or 160),
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {"response": response}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_LLM_PORT", "8765")))
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    if args.preload:
        _load_model()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

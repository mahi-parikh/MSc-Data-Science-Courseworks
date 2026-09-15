import os
import time
import pickle
from functools import lru_cache

import torch
from flask import Flask, render_template, request
from transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,pipeline,)
from peft import PeftModel
from huggingface_hub import hf_hub_download

app = Flask(__name__, template_folder=".")

ARTIFACT_DIR = "./artifacts"

#Sentiment
HF_SENTIMENT_REPO = "Ahila-J/pg10-sentiment"
HF_SENTIMENT_FILENAME = "sklearn_sentiment_pooled.pkl"

#Sarcasm
SARCASM_ADAPTER_REPOS = {
"en-UK": "Ahila-J/pg10-sarcasm-lora-en-uk",
"en-AU": "Ahila-J/pg10-sarcasm-lora-en-au",
"en-IN": "Ahila-J/pg10-sarcasm-lora-en-in",
}

BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = DEVICE == "cuda"


def parse_sarcasm_output(text: str) -> str:
    text = text.strip().lower()
    if "not_sarcastic" in text:
        return "not_sarcastic"
    if "sarcastic" in text:
        return "sarcastic"
    return "not_sarcastic"


@lru_cache(maxsize=1)
def load_sentiment_model():
    model_file = hf_hub_download(
    repo_id=HF_SENTIMENT_REPO,
    filename=HF_SENTIMENT_FILENAME,
    )
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    return model


@lru_cache(maxsize=1)
def load_base_llm():
    kwargs = {}
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_LLM, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_LLM, **kwargs)

    if DEVICE == "cpu":
        model = model.to("cpu")
    model.eval()

    return tokenizer, model


@lru_cache(maxsize=3)
def load_sarcasm_pipeline(variety: str):
    adapter_repo = SARCASM_ADAPTER_REPOS[variety]
    tokenizer, base_model = load_base_llm()
    model = PeftModel.from_pretrained(base_model, adapter_repo)
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
        return_full_text=False,
    )


def predict_sentiment(text: str, variety: str) -> str:
    model = load_sentiment_model()
    pred = int(model.predict([text])[0])
    return "positive" if pred == 1 else "negative"


def predict_sarcasm(text: str, variety: str) -> str:
    gen_pipe = load_sarcasm_pipeline(variety)

    prompt = (
        "Classify the following text for sarcasm.\n"
        "Return only one label: sarcastic or not_sarcastic.\n\n"
        f"Text: {text}\n"
        "Label:"
    )

    output = gen_pipe(prompt)[0]["generated_text"]
    return parse_sarcasm_output(output)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    latency_ms = None

    task = "sarcasm"
    variety = "en-AU"
    text = ""

    if request.method == "POST":
        task = request.form.get("task", "sarcasm")
        variety = request.form.get("variety", "en-AU")
        text = request.form.get("text", "").strip()

        try:
            start = time.perf_counter()

            if task == "sentiment":
                prediction = predict_sentiment(text, variety)
            elif task == "sarcasm":
                prediction = predict_sarcasm(text, variety)
            else:
                raise ValueError("Unknown task selected.")

            latency_ms = round((time.perf_counter() - start) * 1000, 2)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        latency_ms=latency_ms,
        task=task,
        variety=variety,
        text=text,
        device=DEVICE,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
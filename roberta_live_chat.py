import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("GROK_API_KEY"), "GROK_API_KEY not found in .env"

# -----------------------------
# Binning setup
# -----------------------------
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]
ATTENTION_THRESHOLD = 0.35


def bin_signal_value(v: float) -> int:
    v = float(np.clip(v, 0.0, 1.0))
    if v < 0.05:
        return 0
    elif v < 0.15:
        return 1
    elif v < 0.45:
        return 2
    elif v < 0.70:
        return 3
    else:
        return 4


def describe_bin(bin_id: int) -> str:
    return BIN_LABELS[bin_id]


def classify_signal_delta(prev_val: float, curr_val: float) -> str:
    delta = curr_val - prev_val
    if delta > 0.15:
        return "↑ rising sharply"
    elif delta > 0.07:
        return "↑ rising"
    elif delta < -0.15:
        return "↓ falling sharply"
    elif delta < -0.07:
        return "↓ falling"
    else:
        return "→ stable"


# -----------------------------
# Model
# -----------------------------
class DistilRoBERTaRegressor(nn.Module):
    def __init__(self, model_name, num_signals):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.output_layer = nn.Linear(self.encoder.config.hidden_size, num_signals)

    def masked_mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.output_layer(pooled)


def load_model(path, device):
    artifact = torch.load(path, map_location=device)
    model = DistilRoBERTaRegressor(
        artifact["model_name"],
        len(artifact["signals"])
    ).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(artifact["model_name"])

    return model, tokenizer, artifact["signals"], artifact["max_length"]


def predict(text, model, tokenizer, max_length, device):
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        out = model(enc["input_ids"].to(device),
                    enc["attention_mask"].to(device))

    return out.cpu().numpy()[0]


def build_signals(signals, raw):
    rows = []
    for s, v in zip(signals, np.clip(raw, 0, 1)):
        rows.append({
            "signal": s,
            "value": float(v),
            "bin": describe_bin(bin_signal_value(v)),
            "attended": v >= ATTENTION_THRESHOLD
        })
    return rows


def build_deltas(prev, curr):
    prev_map = {r["signal"]: r["value"] for r in prev}
    out = []
    for r in curr:
        p = prev_map.get(r["signal"], r["value"])
        out.append({
            "signal": r["signal"],
            "prev": p,
            "curr": r["value"],
            "dir": classify_signal_delta(p, r["value"])
        })
    return out


def build_prompt(text, signals, deltas, history):
    hist = "\n".join([f'User: {h["user"]}\nAssistant: {h["assistant"]}'
                      for h in history]) if history else "[First message]"

    sig = "\n".join([f"{r['signal']}: {r['bin']} ({r['value']:.3f})"
                     + (" [ATTEND]" if r["attended"] else "")
                     for r in signals])

    delt = "\n".join([f"{d['signal']}: {d['prev']:.3f} → {d['curr']:.3f} {d['dir']}"
                      for d in deltas])

    return f"""
Conversation History:
{hist}

User:
{text}

Signals:
{sig}

Deltas:
{delt}

Respond naturally with emotional awareness, trajectory, and memory.
"""


# -----------------------------
# Grok
# -----------------------------
client = OpenAI(
    api_key=os.environ["GROK_API_KEY"],
    base_url="https://api.x.ai/v1"
)


def call_grok(prompt):
    try:
        res = client.responses.create(
            model="grok-4-1-fast-reasoning",  # ← correct model format for this API
            input=[{
                "role": "user",
                "content": prompt
            }],
            max_output_tokens=512,
            temperature=0.7
        )

        # Preferred shortcut
        if hasattr(res, "output_text") and res.output_text:
            return res.output_text.strip()

        # Fallback (more robust parsing)
        return res.output[0].content[0].text.strip()

    except Exception as e:
        return f"[Error: {e}]"


# -----------------------------
# MULTILINE INPUT (DOUBLE ENTER)
# -----------------------------
def get_multiline_input():
    print("You (press Enter twice to send):")
    lines = []

    while True:
        line = input()
        if line == "":
            if lines:  # second enter
                break
        else:
            lines.append(line)

    return "\n".join(lines)


# -----------------------------
# LIVE CHAT
# -----------------------------
def run_chat(model, tokenizer, signals, max_length, device):
    history = []
    prev = None

    print("\n=== Emotion-Aware Grok Chat ===\n")

    # -----------------------------
    # FIRST MESSAGE (assistant)
    # -----------------------------
    intro = "Hey — I’ll track how you're feeling as we talk, not just what you say. What's been on your mind?"
    print(f"Grok: {intro}\n")

    history.append({
        "user": "[SYSTEM INIT]",
        "assistant": intro
    })

    while True:
        user = get_multiline_input()

        if user.lower() in ["exit", "quit"]:
            break

        raw = predict(user, model, tokenizer, max_length, device)
        sig = build_signals(signals, raw)

        delt = build_deltas(prev, sig) if prev else build_deltas(sig, sig)

        prompt = build_prompt(user, sig, delt, history)

        # -----------------------------
        # PRINT PROMPT
        # -----------------------------
        print("\n" + "="*80)
        print("PROMPT SENT TO GROK")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

        try:
            reply = call_grok(prompt)
        except Exception as e:
            reply = f"[Error: {e}]"

        print(f"\nGrok: {reply}\n")

        history.append({"user": user, "assistant": reply})
        prev = sig


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, signals, max_length = load_model(
        "models/distilroberta_regressor_full.pt",
        device
    )

    run_chat(model, tokenizer, signals, max_length, device)
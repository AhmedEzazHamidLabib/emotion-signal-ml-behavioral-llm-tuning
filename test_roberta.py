import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Binning setup
# -----------------------------
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]


def bin_signal_value(v: float) -> int:
    """
    Convert a continuous signal value into an integer bin ID.

    Bins:
        0 = none      : [0.00, 0.05)
        1 = very_low  : [0.05, 0.15)
        2 = low       : [0.15, 0.45)
        3 = medium    : [0.45, 0.70)
        4 = high      : [0.70, 1.00]
    """
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
    """
    Convert a bin ID into a human-readable label.
    """
    return BIN_LABELS[bin_id]


# -----------------------------
# DistilRoBERTa model definition
# -----------------------------
class DistilRoBERTaRegressor(nn.Module):
    def __init__(self, model_name, num_signals, dropout=0.2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_signals)

    def masked_mean_pooling(self, last_hidden_state, attention_mask):
        """
        Mean-pool token embeddings while ignoring padding tokens.

        Inputs:
            last_hidden_state : [B, T, H]
            attention_mask    : [B, T]

        Output:
            pooled            : [B, H]
        """
        mask = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
        masked_hidden = last_hidden_state * mask

        summed = masked_hidden.sum(dim=1)             # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)      # [B, 1]

        pooled = summed / counts
        return pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state   # [B, T, H]
        pooled = self.masked_mean_pooling(last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        signals = self.output_layer(pooled)             # [B, num_signals]
        return signals


# -----------------------------
# Model loading
# -----------------------------
def load_distilroberta_model(model_path: str, device: torch.device):
    """
    Load the saved DistilRoBERTa artifact and reconstruct both
    the model and tokenizer.
    """
    artifact = torch.load(model_path, map_location=device)

    model_name = artifact["model_name"]
    signals = artifact["signals"]
    max_length = artifact["max_length"]

    model = DistilRoBERTaRegressor(
        model_name=model_name,
        num_signals=len(signals),
        dropout=0.2
    ).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, signals, max_length, model_name


# -----------------------------
# Single-text inference
# -----------------------------
def predict_single_text(text: str, model, tokenizer, max_length, device):
    """
    Run inference on a single text input and return raw predictions.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    return predictions.cpu().numpy()[0]


# -----------------------------
# Reporting
# -----------------------------
def print_prediction_report(text: str, model, tokenizer, signals, max_length, device):
    """
    Generate and print a detailed signal prediction report for one text input.
    """
    print("\n" + "=" * 80)
    print("INPUT TEXT FOR INFERENCE")
    print("=" * 80)
    print(text)

    raw_pred = predict_single_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device
    )

    clipped_pred = np.clip(raw_pred, 0.0, 1.0)

    rows = []
    for sig, raw_v, clip_v in zip(signals, raw_pred, clipped_pred):
        bin_id = bin_signal_value(clip_v)
        bin_name = describe_bin(bin_id)

        rows.append({
            "signal": sig,
            "raw": float(raw_v),
            "clipped": float(clip_v),
            "bin_id": bin_id,
            "bin": bin_name
        })

    print("\n" + "=" * 80)
    print("DETAILED SIGNAL PREDICTIONS (RAW, CLIPPED, AND BINNED)")
    print("=" * 80)
    print(f"{'Signal':20s} {'Raw':>10s} {'Clipped':>10s} {'Bin':>10s}")
    print("-" * 80)
    for row in rows:
        print(f"{row['signal']:20s} {row['raw']:10.4f} {row['clipped']:10.4f} {row['bin']:>10s}")

    rows_sorted = sorted(rows, key=lambda r: r["clipped"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 STRONGEST PREDICTED SIGNALS")
    print("=" * 80)
    for row in rows_sorted[:5]:
        print(f"{row['signal']:20s} {row['clipped']:.4f} ({row['bin']})")

    salient = [r for r in rows_sorted if r["bin_id"] >= 3]

    print("\n" + "=" * 80)
    print("SALIENT SIGNALS (MEDIUM OR HIGH INTENSITY)")
    print("=" * 80)
    if salient:
        for row in salient:
            print(f"{row['signal']:20s} {row['clipped']:.4f} ({row['bin']})")
    else:
        print("No signals reached medium or high intensity.")

    print("\n" + "=" * 80)
    print("SIGNAL VALUE DICTIONARY (CLIPPED CONTINUOUS VALUES)")
    print("=" * 80)
    signal_dict = {row["signal"]: round(row["clipped"], 4) for row in rows}
    print(json.dumps(signal_dict, indent=2))

    print("\n" + "=" * 80)
    print("SIGNAL BIN DICTIONARY (DISCRETE INTENSITY LABELS)")
    print("=" * 80)
    bin_dict = {row["signal"]: row["bin"] for row in rows}
    print(json.dumps(bin_dict, indent=2))


# -----------------------------
# Optional: simple reusable API
# -----------------------------
def predict_as_dict(text: str, model, tokenizer, signals, max_length, device):
    """
    Return a dictionary of clipped continuous predictions.
    Useful if you want to call this from another Python module.
    """
    raw_pred = predict_single_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device
    )

    clipped_pred = np.clip(raw_pred, 0.0, 1.0)
    return {sig: float(v) for sig, v in zip(signals, clipped_pred)}


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer, signals, max_length, model_name = load_distilroberta_model(
        "models/distilroberta_regressor_full.pt",
        device
    )

    print(f"Loaded model      : {model_name}")
    print(f"Max token length  : {max_length}")
    print(f"Number of signals : {len(signals)}")

    test_text = "I AM SO ANGRY AT YOU THAT I CANT BELIEVE WHAT TO DO"

    print_prediction_report(
        text=test_text,
        model=model,
        tokenizer=tokenizer,
        signals=signals,
        max_length=max_length,
        device=device
    )

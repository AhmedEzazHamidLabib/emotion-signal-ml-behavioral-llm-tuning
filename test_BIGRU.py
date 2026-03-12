import json
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Binning setup
# -----------------------------
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]


def bin_signal_value(v: float) -> int:
    """
    Convert a continuous signal value into an integer bin ID.
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
# Tokenization
# -----------------------------
# Must match training-time tokenization exactly.
def tokenize(text: str):
    return text.lower().split()


def encode_text(text: str, stoi: dict, unk_idx: int):
    """
    Convert raw text into a list of token IDs using the saved vocabulary.
    """
    tokens = tokenize(text)
    return [stoi.get(token, unk_idx) for token in tokens]


# -----------------------------
# BiGRU model definition
# -----------------------------
class BiGRURegressor(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_signals,
        padding_idx
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.output_layer = nn.Linear(
            hidden_dim * 2,
            num_signals
        )

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.gru(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        mask = (input_ids != 0).unsqueeze(-1).float()
        mask = mask[:, :output.size(1), :]

        masked_output = output * mask
        summed = masked_output.sum(dim=1)
        lengths = lengths.unsqueeze(1).float()

        sentence_vector = summed / lengths
        signals = self.output_layer(sentence_vector)

        return signals


# -----------------------------
# Model loading
# -----------------------------
def load_bigru_model(model_path: str, device: torch.device):
    """
    Load the saved BiGRU artifact and reconstruct the model.
    """
    artifact = torch.load(model_path, map_location=device)

    stoi = artifact["vocab"]
    itos = artifact["itos"]
    signals = artifact["signals"]
    embedding_dim = artifact["embedding_dim"]
    hidden_dim = artifact["hidden_dim"]
    pad_idx = artifact["pad_idx"]
    unk_idx = artifact["unk_idx"]

    vocab_size = len(stoi)
    num_signals = len(signals)

    model = BiGRURegressor(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_signals=num_signals,
        padding_idx=pad_idx
    ).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    return model, stoi, itos, signals, pad_idx, unk_idx


# -----------------------------
# Single-text inference
# -----------------------------
def predict_single_text(text: str, model, stoi, unk_idx, device):
    """
    Run inference on a single text input and return raw predictions.
    """
    encoded = encode_text(text, stoi, unk_idx)

    # Avoid empty input crashing the GRU pipeline
    if len(encoded) == 0:
        encoded = [unk_idx]

    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    lengths = torch.tensor([len(encoded)], dtype=torch.long).to(device)

    with torch.no_grad():
        predictions = model(input_ids, lengths)

    return predictions.cpu().numpy()[0]


# -----------------------------
# Reporting
# -----------------------------
def print_prediction_report(text: str, model, stoi, unk_idx, signals, device):
    """
    Generate and print a detailed signal prediction report for one text input.
    """
    print("\n" + "=" * 80)
    print("INPUT TEXT FOR INFERENCE")
    print("=" * 80)
    print(text)

    raw_pred = predict_single_text(text, model, stoi, unk_idx, device)
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
# Main execution
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, stoi, itos, signals, pad_idx, unk_idx = load_bigru_model(
        "models/bigru_regressor_full.pt",
        device
    )

    test_text = "I AM SO ANGRY AT YOU THAT I CANT BELIEVE WHAT TO DO"

    print_prediction_report(
        text=test_text,
        model=model,
        stoi=stoi,
        unk_idx=unk_idx,
        signals=signals,
        device=device
    )
import json
import os
import numpy as np
import torch
import torch.nn as nn
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env"
assert os.environ.get("GROK_API_KEY"), "GROK_API_KEY not found in .env"

# -----------------------------
# Binning setup
# -----------------------------
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]


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


# -----------------------------
# Tokenization
# -----------------------------
def tokenize(text: str):
    return text.lower().split()


def encode_text(text: str, stoi: dict, unk_idx: int):
    tokens = tokenize(text)
    return [stoi.get(token, unk_idx) for token in tokens]


# -----------------------------
# BiGRU model definition
# -----------------------------
class BiGRURegressor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_signals, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.output_layer = nn.Linear(hidden_dim * 2, num_signals)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        mask = (input_ids != 0).unsqueeze(-1).float()
        mask = mask[:, :output.size(1), :]
        masked_output = output * mask
        summed = masked_output.sum(dim=1)
        lengths = lengths.unsqueeze(1).float()
        sentence_vector = summed / lengths
        return self.output_layer(sentence_vector)


# -----------------------------
# Model loading
# -----------------------------
def load_bigru_model(model_path: str, device: torch.device):
    artifact = torch.load(model_path, map_location=device)
    stoi = artifact["vocab"]
    itos = artifact["itos"]
    signals = artifact["signals"]
    embedding_dim = artifact["embedding_dim"]
    hidden_dim = artifact["hidden_dim"]
    pad_idx = artifact["pad_idx"]
    unk_idx = artifact["unk_idx"]

    model = BiGRURegressor(
        vocab_size=len(stoi),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_signals=len(signals),
        padding_idx=pad_idx
    ).to(device)

    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, stoi, itos, signals, pad_idx, unk_idx


# -----------------------------
# Single-text inference
# -----------------------------
def predict_single_text(text, model, stoi, unk_idx, device):
    encoded = encode_text(text, stoi, unk_idx)
    if len(encoded) == 0:
        encoded = [unk_idx]
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    lengths = torch.tensor([len(encoded)], dtype=torch.long).to(device)
    with torch.no_grad():
        predictions = model(input_ids, lengths)
    return predictions.cpu().numpy()[0]


# -----------------------------
# Build signal summary for LLM prompt
# -----------------------------
def build_signal_summary(signals, raw_pred):
    clipped = np.clip(raw_pred, 0.0, 1.0)
    rows = []
    for sig, v in zip(signals, clipped):
        bin_id = bin_signal_value(v)
        rows.append({
            "signal": sig,
            "value": round(float(v), 4),
            "bin": describe_bin(bin_id)
        })
    return rows


def format_signals_for_prompt(signal_rows):
    lines = [f"  {r['signal']}: {r['bin']} ({r['value']:.4f})" for r in signal_rows]
    return "\n".join(lines)


# -----------------------------
# Prompt builders
# -----------------------------
def build_prompt_conditioned(text: str, signal_rows: list) -> str:
    signal_block = format_signals_for_prompt(signal_rows)
    return f"""You are an empathetic conversational assistant. A user sent the following message:

"{text}"

An emotion analysis model predicted the following psychological signal intensities for this message:

{signal_block}

Using these signals to inform your response strategy, reply to the user in a way that is
appropriate to their emotional state. Be specific — let the signal levels guide your tone,
level of validation, urgency of support, and choice of language.

After your response, include a section titled "Rationale for Response Strategy" that explains
signal by signal (focusing on the most influential ones) how each signal intensity shaped your
response. Be specific about which signals drove which choices in tone, wording, and strategy."""


def build_prompt_no_signals(text: str) -> str:
    return f"""You are an empathetic conversational assistant. A user sent the following message:

"{text}"

Reply to the user in a way that is appropriate to their emotional state. Be specific in your
tone, level of validation, urgency of support, and choice of language.

After your response, include a section titled "Rationale for Response Strategy" that explains
how you interpreted the emotional state of the message and how that shaped your response."""


# -----------------------------
# LLM calls
# -----------------------------
def call_gpt(prompt: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def call_grok(prompt: str) -> str:
    client = OpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model="grok-3-fast",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# JSON output
# -----------------------------
def save_results_to_json(results: list, path: str = "bigru_test_results_2.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {path}")


# -----------------------------
# Reporting
# -----------------------------
def print_prediction_report(text, model, stoi, unk_idx, signals, device):
    print("\n" + "=" * 80)
    print("INPUT TEXT")
    print("=" * 80)
    print(text)

    raw_pred = predict_single_text(text, model, stoi, unk_idx, device)
    signal_rows = build_signal_summary(signals, raw_pred)

    print("\n" + "=" * 80)
    print("SIGNAL PREDICTIONS")
    print("=" * 80)
    print(f"{'Signal':20s} {'Value':>10s} {'Bin':>10s}")
    print("-" * 80)
    for r in signal_rows:
        print(f"{r['signal']:20s} {r['value']:10.4f} {r['bin']:>10s}")

    salient = [r for r in signal_rows if bin_signal_value(r['value']) >= 3]
    print("\n" + "=" * 80)
    print("SALIENT SIGNALS (MEDIUM OR HIGH)")
    print("=" * 80)
    if salient:
        for r in salient:
            print(f"  {r['signal']:20s} {r['value']:.4f} ({r['bin']})")
    else:
        print("  None reached medium or high intensity.")

    # --- Signal-conditioned prompts ---
    conditioned_prompt = build_prompt_conditioned(text, signal_rows)

    print("\n" + "=" * 80)
    print("GPT-4o RESPONSE (signal-conditioned)")
    print("=" * 80)
    gpt_conditioned = ""
    try:
        gpt_conditioned = call_gpt(conditioned_prompt)
        print(gpt_conditioned)
    except Exception as e:
        gpt_conditioned = f"GPT call failed: {e}"
        print(gpt_conditioned)

    print("\n" + "=" * 80)
    print("GROK RESPONSE (signal-conditioned)")
    print("=" * 80)
    grok_conditioned = ""
    try:
        grok_conditioned = call_grok(conditioned_prompt)
        print(grok_conditioned)
    except Exception as e:
        grok_conditioned = f"Grok call failed: {e}"
        print(grok_conditioned)

    # --- Unconditioned prompts (no signals) ---
    bare_prompt = build_prompt_no_signals(text)

    print("\n" + "=" * 80)
    print("GPT-4o RESPONSE (no signals)")
    print("=" * 80)
    gpt_generic = ""
    try:
        gpt_generic = call_gpt(bare_prompt)
        print(gpt_generic)
    except Exception as e:
        gpt_generic = f"GPT call failed: {e}"
        print(gpt_generic)

    print("\n" + "=" * 80)
    print("GROK RESPONSE (no signals)")
    print("=" * 80)
    grok_generic = ""
    try:
        grok_generic = call_grok(bare_prompt)
        print(grok_generic)
    except Exception as e:
        grok_generic = f"Grok call failed: {e}"
        print(grok_generic)

    # --- Build result record ---
    result = {
        "input_text": text,
        "signal_predictions": {
            r["signal"]: {
                "value": r["value"],
                "bin": r["bin"]
            } for r in signal_rows
        },
        "salient_signals": {
            r["signal"]: {
                "value": r["value"],
                "bin": r["bin"]
            } for r in salient
        },
        "signal_conditioned_responses": {
            "gpt4o": gpt_conditioned,
            "grok": grok_conditioned
        },
        "generic_responses": {
            "gpt4o": gpt_generic,
            "grok": grok_generic
        }
    }

    return result


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, stoi, itos, signals, pad_idx, unk_idx = load_bigru_model(
        "models/bigru_regressor_full.pt",
        device
    )

    test_sentences = [
        "I've been smiling and nodding through every conversation at work lately because explaining how I actually feel would take more energy than I have, and honestly I'm not even sure I could put it into words right now, but I noticed today that I've started dreading my phone lighting up.",

        "My manager praised my work in front of the whole team today and said it was exactly what the project needed, and everyone looked at me like I should feel proud, and I did smile, but on the way home I kept thinking about the three assumptions I made that nobody checked and what happens if they're wrong.",

        "I reached out to an old friend after almost two years of silence because I genuinely missed them, and they responded warmly and said we should catch up soon, and I said yes absolutely, and then I put my phone down and realized I felt more alone after the message than before I sent it.",

        "I keep telling myself I'm fine with how things ended because it was the right decision and I made it with a clear head, but I've noticed I change the subject every time someone brings up that city, and last week I dreamed about it and woke up more tired than when I went to sleep.",

        "Everyone at the party kept coming up to me and including me in conversations and laughing at things I said, and by any measure it was a good night, but I drove home feeling like I had been performing for four hours straight and the version of me that showed up tonight wasn't quite the real one.",

        "I finished the project exactly the way the client asked and delivered it on time and they said it was great and asked if I was available for the next one, and I said yes and sent a professional email, and then I sat at my desk for a while not really doing anything because I couldn't remember why any of it mattered.",

        "I've started leaving the office a few minutes late every day not because I have more work to do but because the walk home is the only part of the day where nobody needs anything from me and I can just be in my own head for a little while, and I've started to notice how much I look forward to that walk.",

        "I told my therapist I'd had a pretty stable week and that was true in the sense that nothing bad happened, but I also didn't mention that I spent most of Sunday lying on the couch watching things I'd already seen because starting something new felt like too much of a commitment, and I'm still not sure why I left that part out.",

        "My partner did something really thoughtful for me last weekend, planned the whole day around things I like, and I was genuinely happy in the moment, but later that night I found myself feeling a strange kind of sadness I couldn't explain, like I was aware of how good things were and somehow that made me more anxious, not less.",

        "I got the email saying my application had moved to the final round and I immediately texted three people who would want to know, and they all responded with excitement, and I matched their energy in my replies, and then I closed my phone and sat with this very quiet, very specific fear that I might actually get it.",
    ]

    all_results = []

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'#' * 80}")
        print(f"TEST CASE {i} / {len(test_sentences)}")
        print(f"{'#' * 80}")
        result = print_prediction_report(
            text=sentence,
            model=model,
            stoi=stoi,
            unk_idx=unk_idx,
            signals=signals,
            device=device
        )
        result["test_case"] = i
        all_results.append(result)

        # Save after every sentence so you don't lose progress if it crashes
        save_results_to_json(all_results)
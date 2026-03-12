from collections import Counter
import numpy as np
from datasets import load_from_disk
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
from utils import load_signals

import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load the ordered list of emotional signal names defined in emotion_prototypes.json.
# These signals represent the continuous emotional dimensions (e.g., valence,
# arousal, threat, curiosity, etc.) that the model predicts. The ordering is
# important because it defines the fixed index positions used throughout the
# dataset, training pipeline, and evaluation code.
SIGNALS = load_signals("emotion_prototypes.json")
# -----------------------------
# 1) Load annotated dataset splits
# -----------------------------
# The GoEmotions dataset has previously been processed and annotated with
# continuous emotional signal vectors. Each example now contains:
#
#   text : the original Reddit comment
#   y    : a continuous vector representing the emotional signal values
#
# The dataset has been saved to disk in three standard machine learning splits:
#
#   train       → used to train the model parameters
#   validation  → used to evaluate model performance during development
#   test        → used only for final evaluation after the model is fixed
#
# We load each split from disk using HuggingFace's `load_from_disk`,
# which restores the dataset in its original Arrow format without
# recomputing the annotations.
train_ds = load_from_disk("data/go_emotions_annotated_train")
val_ds = load_from_disk("data/go_emotions_annotated_validation")
test_ds = load_from_disk("data/go_emotions_annotated_test")

# -----------------------------
# 2) Extract input texts and target signal vectors
# -----------------------------
# Each dataset example contains two relevant fields:
#
#   text : the original Reddit comment (model input)
#   y    : the annotated continuous emotional signal vector (model target)
#
#
# The target field `y` contains the continuous emotional signal values
# corresponding to each text. These are converted into NumPy arrays
# so they can be used directly by the regression model.
#
# Each row in y represents one example, and each column corresponds
# to a specific emotional signal (ordered according to SIGNALS).
#
# The dtype is explicitly set to float32 to reduce memory usage and
# ensure compatibility with numerical operations during training
# and evaluation.
X_train_text = train_ds["text"]
X_val_text = val_ds["text"]
X_test_text = test_ds["text"]

y_train = np.array(train_ds["y"], dtype=np.float32)
y_val = np.array(val_ds["y"], dtype=np.float32)
y_test = np.array(test_ds["y"], dtype=np.float32)

# =============================================================================
# Tokenization and Vocabulary Construction
# =============================================================================
# The BiGRU model cannot consume raw text directly.
# It requires each sentence to be converted into a sequence of integer token IDs.
#
# This step performs:
#   1) tokenization of raw text into word-level tokens
#   2) construction of a training vocabulary
#   3) mapping from token -> integer ID
#   4) conversion of dataset texts into lists of token IDs
# =============================================================================

# -----------------------------
# Define a simple tokenizer
# -----------------------------
# For the first version of the BiGRU, we use a simple whitespace + lowercase
# tokenizer. This keeps the pipeline easy to understand and debug.
#
# Example:
#   "I Feel Empty!" -> ["i", "feel", "empty!"]
#
# Later, if desired, this can be replaced with a better tokenizer.
def tokenize(text: str):
    return text.lower().split()

# -----------------------------
# Special tokens
# -----------------------------
# <PAD> is used later when batching sequences of different lengths.
# <UNK> is used for words that were not seen in the training vocabulary.
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1


# -----------------------------
# Build vocabulary from training text only
# -----------------------------
# IMPORTANT:
# The vocabulary must be built only from the training split.
# We do not use validation/test text to build the vocabulary, because that
# would leak information from evaluation data into training.
token_counter = Counter()

for text in X_train_text:
    tokens = tokenize(text)
    token_counter.update(tokens)

# Vocabulary filtering:
# Keep words that appear at least min_freq times in the training set.
# This reduces noise from extremely rare words.
min_freq = 2

vocab_tokens = [
    token for token, count in token_counter.items()
    if count >= min_freq
]

# Sort vocabulary for reproducibility
vocab_tokens = sorted(vocab_tokens)

# Create token -> id mapping
stoi = {
    PAD_TOKEN: PAD_IDX,
    UNK_TOKEN: UNK_IDX,
}

for i, token in enumerate(vocab_tokens, start=2):
    stoi[token] = i

# Create id -> token mapping
itos = {idx: token for token, idx in stoi.items()}

# -----------------------------
# Convert text into token IDs
# -----------------------------
# Any token not found in the training vocabulary is mapped to UNK_IDX.
def encode_text(text: str):
    tokens = tokenize(text)
    return [stoi.get(token, UNK_IDX) for token in tokens]


# Encode all dataset splits
X_train_ids = [encode_text(text) for text in X_train_text]
X_val_ids = [encode_text(text) for text in X_val_text]
X_test_ids = [encode_text(text) for text in X_test_text]

# =============================================================================
# Dataset and DataLoader Construction
# =============================================================================
# The BiGRU model is trained in batches rather than one example at a time.
# However, text sequences have variable lengths, so we need a custom pipeline
# that can:
#
#   1) return one encoded example and its target vector
#   2) pad sequences within each batch to the same length
#   3) keep track of the original lengths before padding
#
# This is necessary because the embedding + GRU model expects batched tensors,
# while the raw encoded data is currently stored as Python lists of different
# lengths.
# =============================================================================

# -----------------------------
# Custom Dataset
# -----------------------------
# This class wraps the encoded texts and target vectors so PyTorch can access
# them one example at a time.
#
# Each returned example contains:
#   input_ids : token ID sequence for one text example
#   target    : continuous emotional signal vector for that example
class EmotionSignalDataset(Dataset):
    def __init__(self, encoded_texts, targets):
        self.encoded_texts = encoded_texts
        self.targets = targets

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }


# -----------------------------
# Collate function
# -----------------------------
# DataLoader gathers multiple examples into a batch, but because the input
# sequences have different lengths, they cannot be stacked directly.
#
# This function:
#   1) collects all sequences in the batch
#   2) records their original lengths
#   3) pads them to the longest sequence in the batch using PAD_IDX
#   4) stacks the target vectors
#
# The result is a proper rectangular tensor batch that the BiGRU can read.
def collate_batch(batch):
    input_ids_list = [item["input_ids"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])

    lengths = torch.tensor([len(seq) for seq in input_ids_list], dtype=torch.long)

    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return {
        "input_ids": padded_input_ids,   # shape: [B, T]
        "lengths": lengths,              # shape: [B]
        "targets": targets               # shape: [B, num_signals]
    }


# -----------------------------
# Create Dataset objects
# -----------------------------
train_dataset = EmotionSignalDataset(X_train_ids, y_train)
val_dataset = EmotionSignalDataset(X_val_ids, y_val)
test_dataset = EmotionSignalDataset(X_test_ids, y_test)


# -----------------------------
# Create DataLoaders
# -----------------------------
# shuffle=True for training so the model sees examples in different orders.
# shuffle=False for validation and test because order does not matter there.
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)


# =============================================================================
# BiGRU Regressor Model
# =============================================================================
# This model predicts continuous emotional signal values from text.
#
# Architecture:
#
#   input_ids
#       ↓
#   Embedding layer
#       ↓
#   Bidirectional GRU
#       ↓
#   Mean pooling across time
#       ↓
#   Linear projection
#       ↓
#   Predicted emotional signals
# =============================================================================
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

        mask = (input_ids != PAD_IDX).unsqueeze(-1).float()
        mask = mask[:, :output.size(1), :]

        masked_output = output * mask
        summed = masked_output.sum(dim=1)
        lengths = lengths.unsqueeze(1).float()
        sentence_vector = summed / lengths

        signals = self.output_layer(sentence_vector)
        return signals

# =============================================================================
# Model Initialization and Training Setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

vocab_size = len(stoi)
embedding_dim = 200
hidden_dim = 128
num_signals = len(SIGNALS)

model = BiGRURegressor(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_signals=num_signals,
    padding_idx=PAD_IDX
).to(device)

# Loss for continuous signal regression
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n====================")
print("Model Configuration")
print("====================")
print("Vocabulary size :", vocab_size)
print("Embedding dim   :", embedding_dim)
print("Hidden dim      :", hidden_dim)
print("Num signals     :", num_signals)
print("Loss function   : MSELoss")
print("Optimizer       : Adam")

# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, lengths)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_dataset(model, dataloader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            predictions = model(input_ids, lengths)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return all_preds, all_targets

# =============================================================================
# Model Training
# =============================================================================

num_epochs = 10
best_val_mae = float("inf")
best_model_path = "models/bigru_regressor_best.pt"

train_loss_history = []
val_mae_history = []

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    y_val_pred, y_val_true_check = predict_dataset(model, val_loader, device)

    val_mae = mean_absolute_error(y_val_true_check, y_val_pred)

    train_loss_history.append(train_loss)
    val_mae_history.append(val_mae)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val MAE   : {val_mae:.6f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated and saved to {best_model_path}")

print("\nTraining complete.")
print(f"Best validation MAE: {best_val_mae:.6f}")

# =============================================================================
# Load Best Model
# =============================================================================

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print(f"\nLoaded best model from: {best_model_path}")

# =============================================================================
# Binning Helpers
# =============================================================================

BIN_LABELS = ["none", "very_low", "low", "medium", "high"]

def bin_signal_values(arr: np.ndarray) -> np.ndarray:
    """
    Convert continuous values in [0, 1] to discrete bin IDs.

    Bins:
        0 = none      : [0.00, 0.05)
        1 = very_low  : [0.05, 0.15)
        2 = low       : [0.15, 0.45)
        3 = medium    : [0.45, 0.70)
        4 = high      : [0.70, 1.00]
    """
    arr = np.clip(arr, 0.0, 1.0)

    binned = np.zeros_like(arr, dtype=np.int32)
    binned[(arr >= 0.05) & (arr < 0.15)] = 1
    binned[(arr >= 0.15) & (arr < 0.45)] = 2
    binned[(arr >= 0.45) & (arr < 0.70)] = 3
    binned[(arr >= 0.70)] = 4

    return binned

# =============================================================================
# Validation Evaluation - Continuous Metrics
# =============================================================================

y_val_pred, y_val_true = predict_dataset(model, val_loader, device)

val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_r2 = r2_score(y_val_true, y_val_pred)

print("\n====================")
print("Validation Set – Regression Metrics (Continuous Signal Prediction)")
print("====================")

print(
    f"MAE  (Mean Absolute Error): {val_mae:.4f}\n"
    "  Measures the average absolute difference between predicted and true\n"
    "  continuous signal values across all validation examples and signals.\n"
    "  Example:\n"
    "      True signal = 0.70, Predicted = 0.65 → Absolute error = 0.05\n"
    "      True signal = 0.30, Predicted = 0.10 → Absolute error = 0.20\n"
    "  MAE averages these absolute differences.\n"
    "  Interpretation:\n"
    "      Lower MAE means the predicted signal intensities are numerically\n"
    "      closer to the true emotional signal values.\n"
)

print(
    f"RMSE (Root Mean Squared Error): {val_rmse:.4f}\n"
    "  Measures the square root of the average squared error.\n"
    "  Unlike MAE, RMSE penalizes larger mistakes more heavily because errors\n"
    "  are squared before averaging.\n"
    "  Example:\n"
    "      Error = 0.20 → Squared error = 0.04\n"
    "      Error = 0.50 → Squared error = 0.25\n"
    "  Interpretation:\n"
    "      Lower RMSE means the model avoids large prediction mistakes.\n"
)

print(
    f"R^2  (Coefficient of Determination): {val_r2:.4f}\n"
    "  Measures how much of the variation in the true signal values is\n"
    "  explained by the model's predictions.\n"
    "  Interpretation:\n"
    "      R^2 ≈ 1.0 → predictions track the true variation very well\n"
    "      R^2 ≈ 0.0 → model performs similarly to always predicting the mean\n"
    "      R^2 < 0.0 → model is worse than the mean baseline\n"
)

# =============================================================================
# Validation Evaluation - Binned Classification Metrics
# =============================================================================

y_val_pred_clipped = np.clip(y_val_pred, 0.0, 1.0)

y_val_bins_true = bin_signal_values(y_val_true)
y_val_bins_pred = bin_signal_values(y_val_pred_clipped)

y_val_bins_true_flat = y_val_bins_true.reshape(-1)
y_val_bins_pred_flat = y_val_bins_pred.reshape(-1)

val_acc = accuracy_score(y_val_bins_true_flat, y_val_bins_pred_flat)

val_precision_macro = precision_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)

val_recall_macro = recall_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)

val_f1_macro = f1_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)

val_precision_weighted = precision_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="weighted",
    zero_division=0
)

val_recall_weighted = recall_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="weighted",
    zero_division=0
)

val_f1_weighted = f1_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="weighted",
    zero_division=0
)

print("\n====================")
print("Validation Set – Classification Metrics (Signal Intensity Bins)")
print("====================")

print(
    f"Accuracy: {val_acc:.4f}\n"
    "  Measures the proportion of signal predictions where the predicted\n"
    "  intensity bin exactly matches the true bin.\n"
    "  Example:\n"
    "      True = medium, Predicted = medium → correct\n"
    "      True = high,   Predicted = low    → incorrect\n"
    "  Interpretation:\n"
    "      Higher accuracy means the model often predicts the correct\n"
    "      coarse intensity category of each emotional signal.\n"
)

print(
    f"Precision (Macro): {val_precision_macro:.4f}\n"
    "  Precision asks: when the model predicts a given bin, how often is it right?\n"
    "  Macro precision computes precision for each bin separately and averages\n"
    "  them equally, so rare bins matter as much as common bins.\n"
)

print(
    f"Recall (Macro): {val_recall_macro:.4f}\n"
    "  Recall asks: of all true examples belonging to a bin, how many did the\n"
    "  model correctly recover?\n"
    "  Macro recall gives equal importance to every bin.\n"
)

print(
    f"F1 Score (Macro): {val_f1_macro:.4f}\n"
    "  F1 combines precision and recall into one number.\n"
    "  Macro F1 is useful when you want balanced performance across all bins,\n"
    "  including rare signal intensity levels.\n"
)

print(
    f"Precision (Weighted): {val_precision_weighted:.4f}\n"
    "  Weighted precision is similar to macro precision, but bins with more\n"
    "  examples contribute more to the final score.\n"
    "  This reflects performance on the real class distribution.\n"
)

print(
    f"Recall (Weighted): {val_recall_weighted:.4f}\n"
    "  Weighted recall measures how well the model recovers the true bins while\n"
    "  accounting for how common each bin is in the validation data.\n"
)

print(
    f"F1 Score (Weighted): {val_f1_weighted:.4f}\n"
    "  Weighted F1 summarizes overall binned classification quality while taking\n"
    "  class imbalance into account.\n"
)

# =============================================================================
# Per-Signal Validation Metrics
# =============================================================================

print("\n====================")
print("Per-Signal Validation Metrics")
print("====================")

for i, sig in enumerate(SIGNALS):
    sig_mae = mean_absolute_error(y_val_true[:, i], y_val_pred[:, i])
    sig_rmse = np.sqrt(mean_squared_error(y_val_true[:, i], y_val_pred[:, i]))
    sig_r2 = r2_score(y_val_true[:, i], y_val_pred[:, i])

    sig_true_bins = y_val_bins_true[:, i]
    sig_pred_bins = y_val_bins_pred[:, i]

    sig_acc = accuracy_score(sig_true_bins, sig_pred_bins)
    sig_precision = precision_score(sig_true_bins, sig_pred_bins, average="macro", zero_division=0)
    sig_recall = recall_score(sig_true_bins, sig_pred_bins, average="macro", zero_division=0)
    sig_f1 = f1_score(sig_true_bins, sig_pred_bins, average="macro", zero_division=0)

    print(f"\nSignal: {sig}")
    print(f"  MAE      : {sig_mae:.4f}")
    print(f"  RMSE     : {sig_rmse:.4f}")
    print(f"  R^2      : {sig_r2:.4f}")
    print(f"  Accuracy : {sig_acc:.4f}")
    print(f"  Precision: {sig_precision:.4f}")
    print(f"  Recall   : {sig_recall:.4f}")
    print(f"  F1       : {sig_f1:.4f}")


# =============================================================================
# Detailed Validation Classification Report
# =============================================================================

print("\n====================")
print(
    "Validation Set – Detailed Classification Report\n"
    "This report summarizes model performance for each signal intensity bin\n"
    "(none, very_low, low, medium, high) after flattening all signal predictions\n"
    "across all validation samples.\n"
    "\n"
    "Columns:\n"
    "  precision → proportion of predicted instances of a class that are correct\n"
    "  recall    → proportion of true instances of a class correctly identified\n"
    "  f1-score  → harmonic mean of precision and recall\n"
    "  support   → number of true samples belonging to that class\n"
)
print("====================")

print(classification_report(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    target_names=BIN_LABELS,
    zero_division=0
))

# =============================================================================
# Final Test Evaluation
# =============================================================================

y_test_pred, y_test_true = predict_dataset(model, test_loader, device)
y_test_pred_clipped = np.clip(y_test_pred, 0.0, 1.0)

# Continuous metrics
test_mae = mean_absolute_error(y_test_true, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
test_r2 = r2_score(y_test_true, y_test_pred)

# Binned metrics
y_test_bins_true = bin_signal_values(y_test_true)
y_test_bins_pred = bin_signal_values(y_test_pred_clipped)

y_test_bins_true_flat = y_test_bins_true.reshape(-1)
y_test_bins_pred_flat = y_test_bins_pred.reshape(-1)

test_acc = accuracy_score(y_test_bins_true_flat, y_test_bins_pred_flat)
test_precision_macro = precision_score(
    y_test_bins_true_flat,
    y_test_bins_pred_flat,
    average="macro",
    zero_division=0
)
test_recall_macro = recall_score(
    y_test_bins_true_flat,
    y_test_bins_pred_flat,
    average="macro",
    zero_division=0
)
test_f1_macro = f1_score(
    y_test_bins_true_flat,
    y_test_bins_pred_flat,
    average="macro",
    zero_division=0
)

print("\n====================")
print(
    "Final Test Set Results\n"
    "These metrics evaluate model performance on the held-out test dataset.\n"
    "Regression metrics measure numerical accuracy of continuous signal\n"
    "predictions, while classification metrics measure how accurately the\n"
    "model predicts signal intensity categories after binning."
)
print("====================")

print(
    f"MAE                : {test_mae:.4f}\n"
    "  Average absolute difference between predicted and true continuous\n"
    "  emotional signal values on the unseen test set.\n"
)

print(
    f"RMSE               : {test_rmse:.4f}\n"
    "  Root mean squared error on the test set. Penalizes larger mistakes\n"
    "  more strongly than MAE.\n"
)

print(
    f"R^2                : {test_r2:.4f}\n"
    "  Measures how much of the variation in true emotional signals is\n"
    "  explained by the model on unseen data.\n"
)

print(
    f"Accuracy           : {test_acc:.4f}\n"
    "  Proportion of test signal predictions whose intensity bin exactly\n"
    "  matches the true bin.\n"
)

print(
    f"Precision (macro)  : {test_precision_macro:.4f}\n"
    "  Macro-averaged precision across the five signal intensity bins.\n"
)

print(
    f"Recall (macro)     : {test_recall_macro:.4f}\n"
    "  Macro-averaged recall across the five signal intensity bins.\n"
)

print(
    f"F1 (macro)         : {test_f1_macro:.4f}\n"
    "  Balanced summary of precision and recall across bins on the test set.\n"
)

# =============================================================================
# Save Final Model Artifacts
# =============================================================================

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": stoi,
    "itos": itos,
    "signals": SIGNALS,
    "embedding_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "pad_idx": PAD_IDX,
    "unk_idx": UNK_IDX
}, "models/bigru_regressor_full.pt")

print("\nFull BiGRU model artifact saved to base_model/bigru_regressor_full.pt")
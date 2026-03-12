from datasets import load_from_disk
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from utils import load_signals


# =============================================================================
# Reproducibility
# =============================================================================
# Fix all major random seeds so results are more stable across runs.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# =============================================================================
# Load signal definitions
# =============================================================================
# SIGNALS stores the ordered list of emotional signal names used throughout
# the project. The order must match the order of the target dimensions in y.
SIGNALS = load_signals("emotion_prototypes.json")


# =============================================================================
# 1) Load annotated dataset splits
# =============================================================================
# Each dataset example contains:
#
#   text : the original Reddit comment
#   y    : a continuous emotional signal vector in [0, 1]
#
# These were already preprocessed and saved to disk previously.
train_ds = load_from_disk("data/go_emotions_annotated_train")
val_ds = load_from_disk("data/go_emotions_annotated_validation")
test_ds = load_from_disk("data/go_emotions_annotated_test")


# =============================================================================
# 2) Extract raw texts and target vectors
# =============================================================================
# The target matrix y is converted to float32 for PyTorch compatibility and
# reduced memory usage.
X_train_text = list(train_ds["text"])
X_val_text = list(val_ds["text"])
X_test_text = list(test_ds["text"])

y_train = np.array(train_ds["y"], dtype=np.float32)
y_val = np.array(val_ds["y"], dtype=np.float32)
y_test = np.array(test_ds["y"], dtype=np.float32)


# =============================================================================
# 3) Tokenizer setup
# =============================================================================
# We use DistilRoBERTa as the pretrained text encoder.
#
# This tokenizer converts raw text into:
#   - input_ids
#   - attention_mask
#
# Important:
# The tokenizer must match the exact pretrained model checkpoint.
MODEL_NAME = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =============================================================================
# 4) Pre-tokenize all splits once
# =============================================================================
max_length = 128
batch_size = 16

print("\nTokenizing training split...")
train_encodings = tokenizer(
    X_train_text,
    truncation=True,
    padding="max_length",
    max_length=max_length
)

print("Tokenizing validation split...")
val_encodings = tokenizer(
    X_val_text,
    truncation=True,
    padding="max_length",
    max_length=max_length
)

print("Tokenizing test split...")
test_encodings = tokenizer(
    X_test_text,
    truncation=True,
    padding="max_length",
    max_length=max_length
)

print("Tokenization complete.")


# =============================================================================
# 5) Dataset class for pre-tokenized inputs
# =============================================================================
class EmotionSignalTransformerDataset(Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }


# =============================================================================
# 6) Build Dataset / DataLoader objects
# =============================================================================
train_dataset = EmotionSignalTransformerDataset(train_encodings, y_train)
val_dataset = EmotionSignalTransformerDataset(val_encodings, y_val)
test_dataset = EmotionSignalTransformerDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# =============================================================================
# 6) DistilRoBERTa regressor model
# =============================================================================
# Architecture:
#
#   text
#     ↓
#   DistilRoBERTa encoder
#     ↓
#   mean pooling over token representations
#     ↓
#   dropout
#     ↓
#   linear regression head
#     ↓
#   predicted emotional signal vector
#
# Why mean pooling?
# DistilRoBERTa does not provide the same pooled output API as some other
# transformer models, so a common and effective strategy is to mean-pool the
# last hidden states using the attention mask.
class DistilRoBERTaRegressor(nn.Module):
    def __init__(self, model_name, num_signals, dropout=0.2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_signals)

    def masked_mean_pooling(self, last_hidden_state, attention_mask):
        """
        Compute mean pooling over the token embeddings while ignoring padding.

        Inputs:
            last_hidden_state : [B, T, H]
            attention_mask    : [B, T]

        Output:
            pooled            : [B, H]
        """
        mask = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
        masked_hidden = last_hidden_state * mask      # zero out padding tokens

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


# =============================================================================
# 7) Model initialization and optimization setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

num_signals = len(SIGNALS)

model = DistilRoBERTaRegressor(
    model_name=MODEL_NAME,
    num_signals=num_signals,
    dropout=0.2
).to(device)

# Since this is continuous signal prediction, we use mean squared error loss.
criterion = nn.MSELoss()

# AdamW is standard for transformer fine-tuning.
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 3
total_training_steps = len(train_loader) * num_epochs

# A linear learning-rate schedule with warmup is commonly used for transformers.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_training_steps),
    num_training_steps=total_training_steps
)

print("\n====================")
print("Model Configuration")
print("====================")
print("Backbone model :", MODEL_NAME)
print("Max length     :", max_length)
print("Batch size     :", batch_size)
print("Num signals    :", num_signals)
print("Loss function  :", "MSELoss")
print("Optimizer      :", "AdamW")
print("Scheduler      :", "Linear warmup/decay")


# =============================================================================
# 8) Training and prediction helpers
# =============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch_num=None):
    model.train()
    total_loss = 0.0

    print(f"\nStarting training epoch {epoch_num}...")

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                f"[Epoch {epoch_num}] "
                f"Batch {batch_idx + 1}/{len(dataloader)} "
                f"| Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(dataloader)
    print(f"Finished epoch {epoch_num} | Average train loss: {avg_loss:.6f}")

    return avg_loss


def predict_dataset(model, dataloader, device, stage_name="unknown"):
    model.eval()

    print(f"\nRunning prediction for: {stage_name}")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            predictions = model(input_ids, attention_mask)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"[{stage_name}] Batch {batch_idx + 1}/{len(dataloader)} complete")

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print(f"Finished prediction for: {stage_name}")
    return all_preds, all_targets

# =============================================================================
# 9) Model training
# =============================================================================
best_val_mae = float("inf")
best_model_path = "models/distilroberta_regressor_best.pt"

train_loss_history = []
val_mae_history = []

print("\nAbout to begin training loop...")
print(f"Train loader batches: {len(train_loader)}")
print(f"Validation loader batches: {len(val_loader)}")

for epoch in range(num_epochs):
    print(f"\n====================")
    print(f"STARTING EPOCH {epoch + 1}/{num_epochs}")
    print("====================")

    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch_num=epoch + 1
    )

    print(f"[Epoch {epoch + 1}] Training epoch finished. About to run validation prediction...")

    y_val_pred, y_val_true_check = predict_dataset(model, val_loader, device, stage_name=f"val_epoch_{epoch + 1}")

    print(f"[Epoch {epoch + 1}] Validation prediction finished. About to compute MAE...")

    val_mae = mean_absolute_error(y_val_true_check, y_val_pred)

    train_loss_history.append(train_loss)
    val_mae_history.append(val_mae)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val MAE   : {val_mae:.6f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated and saved to {best_model_path}")

print("\nTraining complete.")
print(f"Best validation MAE: {best_val_mae:.6f}")


# =============================================================================
# 10) Load best model before final evaluation
# =============================================================================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print(f"\nLoaded best model from: {best_model_path}")


# =============================================================================
# 11) Binning helpers
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
# 12) Validation evaluation - continuous regression metrics
# =============================================================================
y_val_pred, y_val_true = predict_dataset(model, val_loader, device)

val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_r2 = r2_score(y_val_true, y_val_pred)

print("\n====================")
print("Validation Set – Regression Metrics (Continuous Signal Prediction)")
print("====================")
print(f"MAE  : {val_mae:.4f}")
print(f"RMSE : {val_rmse:.4f}")
print(f"R^2  : {val_r2:.4f}")


# =============================================================================
# 13) Validation evaluation - binned classification metrics
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

print("\n====================")
print("Validation Set – Classification Metrics (Signal Intensity Bins)")
print("====================")
print(f"Accuracy          : {val_acc:.4f}")
print(f"Precision (macro) : {val_precision_macro:.4f}")
print(f"Recall (macro)    : {val_recall_macro:.4f}")
print(f"F1 (macro)        : {val_f1_macro:.4f}")


# =============================================================================
# 14) Per-signal validation metrics
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
# 15) Detailed validation classification report
# =============================================================================
print("\n====================")
print("Validation Set – Detailed Classification Report")
print("====================")

print(classification_report(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    target_names=BIN_LABELS,
    zero_division=0
))


# =============================================================================
# 16) Final test evaluation
# =============================================================================
y_test_pred, y_test_true = predict_dataset(model, test_loader, device)
y_test_pred_clipped = np.clip(y_test_pred, 0.0, 1.0)

test_mae = mean_absolute_error(y_test_true, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
test_r2 = r2_score(y_test_true, y_test_pred)

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
    "The test set represents completely unseen data and therefore provides\n"
    "the most reliable estimate of how well the model generalizes to new text."
)
print("====================")

print(
    f"MAE (Mean Absolute Error) : {test_mae:.4f}\n"
    "  MAE measures the average absolute difference between the predicted\n"
    "  emotional signal values and the true signal values.\n"
    "\n"
    "  Example:\n"
    "      True signal = 0.70, Predicted = 0.60 → Error = 0.10\n"
    "      True signal = 0.20, Predicted = 0.35 → Error = 0.15\n"
    "\n"
    "  Interpretation:\n"
    "      Lower MAE is better.\n"
    "      A small MAE means the predicted signal intensities are numerically\n"
    "      close to the true emotional signal values.\n"
)

print(
    f"RMSE (Root Mean Squared Error) : {test_rmse:.4f}\n"
    "  RMSE measures the square root of the average squared prediction error.\n"
    "  Because errors are squared before averaging, large mistakes are\n"
    "  penalized more strongly than in MAE.\n"
    "\n"
    "  Example:\n"
    "      Error = 0.20 → Squared error = 0.04\n"
    "      Error = 0.50 → Squared error = 0.25\n"
    "\n"
    "  Interpretation:\n"
    "      Lower RMSE is better.\n"
    "      A low RMSE indicates the model avoids large prediction mistakes.\n"
)

print(
    f"R^2 (Coefficient of Determination) : {test_r2:.4f}\n"
    "  R^2 measures how much of the variation in the true emotional signals\n"
    "  is explained by the model's predictions.\n"
    "\n"
    "  Interpretation:\n"
    "      R^2 ≈ 1.0  → predictions closely track the true signal values\n"
    "      R^2 ≈ 0.0  → model performs similar to always predicting the mean\n"
    "      R^2 < 0.0  → model performs worse than the mean baseline\n"
)

print(
    f"Accuracy : {test_acc:.4f}\n"
    "  Accuracy measures how often the predicted signal intensity bin\n"
    "  exactly matches the true bin after continuous predictions are\n"
    "  converted into discrete categories:\n"
    "      none, very_low, low, medium, high\n"
    "\n"
    "  Interpretation:\n"
    "      Higher accuracy means the model frequently predicts the correct\n"
    "      coarse intensity category of each emotional signal.\n"
)

print(
    f"Precision (Macro) : {test_precision_macro:.4f}\n"
    "  Precision answers the question:\n"
    "      When the model predicts a specific intensity bin, how often is it correct?\n"
    "\n"
    "  Macro precision computes precision for each bin independently and\n"
    "  averages them equally, meaning rare bins are treated just as\n"
    "  importantly as common bins.\n"
    "\n"
    "  Interpretation:\n"
    "      Higher precision means the model produces fewer false positives\n"
    "      when predicting signal intensity levels.\n"
)

print(
    f"Recall (Macro) : {test_recall_macro:.4f}\n"
    "  Recall answers the question:\n"
    "      Of all true instances belonging to a bin, how many did the model\n"
    "      correctly identify?\n"
    "\n"
    "  Macro recall gives equal weight to each bin regardless of frequency.\n"
    "\n"
    "  Interpretation:\n"
    "      Higher recall means the model successfully captures most\n"
    "      examples of each emotional signal intensity.\n"
)

print(
    f"F1 Score (Macro) : {test_f1_macro:.4f}\n"
    "  The F1 score combines precision and recall into a single metric\n"
    "  using their harmonic mean.\n"
    "\n"
    "  Interpretation:\n"
    "      A high F1 score indicates the model balances both:\n"
    "          • predicting the correct bins\n"
    "          • capturing most true signal instances\n"
    "\n"
    "  F1 is especially useful when the distribution of intensity bins\n"
    "  is imbalanced.\n"
)


# =============================================================================
# 17) Save full model artifact for inference/deployment
# =============================================================================
# We save:
#   - best model weights
#   - model name
#   - tokenizer name
#   - signal ordering
#   - max length
#
# This makes it easier to rebuild the full inference pipeline later.
torch.save({
    "model_state_dict": model.state_dict(),
    "model_name": MODEL_NAME,
    "signals": SIGNALS,
    "max_length": max_length
}, "models/distilroberta_regressor_full.pt")

print("\nFull DistilRoBERTa model artifact saved to base_model/distilroberta_regressor_full.pt")
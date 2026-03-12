from datasets import load_from_disk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
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
import joblib

# =============================================================================
# Baseline Model: TF-IDF + Ridge Regression
# =============================================================================
# This script trains a baseline model that predicts a continuous emotional
# signal vector from text input.
#
# The model uses a TF-IDF representation of the text and a Ridge regression
# estimator to produce multi-output continuous predictions for all signals.
#
# Model evaluation is performed using two approaches:
#
# 1) Regression evaluation on the raw continuous predictions:
#       - Mean Absolute Error (MAE)
#       - Root Mean Squared Error (RMSE)
#       - Coefficient of Determination (R²)
#
# 2) Discrete evaluation after converting each predicted signal value
#    into an intensity bin:
#       none, very_low, low, medium, high
#
#    Standard classification metrics are then computed on the binned outputs:
#       - Accuracy
#       - Precision
#       - Recall
#       - F1 score
#
# This model serves as a baseline for predicting emotional signal vectors
# from text before evaluating more complex modeling approaches.
# =============================================================================
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
# The text field is extracted as the input feature set for the model.
# These raw strings will later be converted into numeric feature vectors
# using the TF-IDF vectorizer inside the modeling pipeline.
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

# Print the shapes of the target matrices to verify dataset sizes
# and confirm that each example has the expected number of signals.

print("Train shape:", y_train.shape)
print("Val shape:  ", y_val.shape)
print("Test shape: ", y_test.shape)


# -----------------------------
# 3) Build and train baseline model
# -----------------------------
# The baseline model is implemented as a scikit-learn Pipeline consisting of:
#
# 1) TF-IDF Vectorization
#    Converts raw text into a sparse numerical feature matrix. Each feature
#    represents a word or short phrase extracted from the corpus, weighted by
#    its importance within a document and across the dataset.
#
#    Parameter settings:
#
#      max_features=20000
#          Limits the vocabulary to the 20,000 most informative terms in the
#          corpus. This controls the dimensionality of the feature space and
#          prevents extremely large vocabularies.
#
#      ngram_range=(1, 2)
#          Uses both unigrams (single words) and bigrams (two-word phrases)
#          as features. Including bigrams allows the model to capture simple
#          contextual patterns such as "very happy" or "not good".
#
#      min_df=2
#          Ignores terms that appear in fewer than 2 documents. Very rare
#          words usually contribute little useful information and can add noise.
#
#      max_df=0.95
#          Ignores terms that appear in more than 95% of documents. Extremely
#          common terms provide little discriminative value.
#
#      sublinear_tf=True
#          Applies logarithmic scaling to term frequency so that repeated
#          occurrences of a word within the same document have diminishing
#          impact rather than increasing linearly.
#
# 2) Ridge Regression
#    A linear regression model with L2 regularization used here for
#    multi-output prediction of the continuous emotional signal vector.
#
#      alpha=1.0
#          Regularization strength. Larger values increase the penalty on
#          large coefficient weights, which helps reduce overfitting.
#
# The pipeline ensures that text preprocessing and regression are applied
# consistently during both training and prediction.

baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("ridge", Ridge(alpha=1.0))
])

# Fit the pipeline on the training data.
# The vectorizer learns the vocabulary and TF-IDF statistics from the
# training texts, and the Ridge model learns to map these features
# to the continuous emotional signal vectors.

baseline_model.fit(X_train_text, y_train)

# -----------------------------
# Save trained baseline model
# -----------------------------
joblib.dump(baseline_model, "models/baseline_tfidf_ridge.pkl")
print("\nBaseline model saved to base_model/baseline_tfidf_ridge.pkl")


# -----------------------------
# 4) Generate predictions on the validation set
# -----------------------------
# The trained pipeline is now applied to the validation texts to produce
# predicted emotional signal vectors.
#
# Internally, the pipeline performs two steps automatically:
#
#   1) The TF-IDF vectorizer transforms each validation text into the
#      same feature representation learned from the training data.
#
#   2) The Ridge regression model uses the learned feature weights to
#      compute predicted values for each emotional signal.
#
# The output `y_val_pred` is a matrix where:
#
#   rows    = validation examples
#   columns = emotional signals
#
# Each value represents the predicted continuous intensity of a signal.

y_val_pred = baseline_model.predict(X_val_text)


# -----------------------------
# Clip predictions to valid signal range
# -----------------------------
# Ridge regression is an unconstrained linear model, meaning it can produce
# predictions outside the intended signal range of [0, 1]. For example,
# values slightly below 0 or above 1 may occur due to the linear combination
# of features.
#
# Two versions of the predictions are therefore maintained:
#
#   raw predictions (y_val_pred)
#       Used for regression metrics such as MAE, RMSE, and R² because these
#       metrics measure the model's exact numerical prediction behavior.
#
#   clipped predictions (y_val_pred_clipped)
#       Used for later binning into discrete intensity categories. Clipping
#       ensures that all signal values fall within the valid [0, 1] range
#       before they are converted into bins.

y_val_pred_clipped = np.clip(y_val_pred, 0.0, 1.0)

# -----------------------------
# 5) Compute regression evaluation metrics
# -----------------------------
# These metrics evaluate how closely the predicted continuous signal values
# match the true target values on the validation set.
#
# The comparison is performed using the raw model predictions (not the
# clipped values) because regression metrics measure the numerical
# accuracy of the model's outputs.
#
# MAE (Mean Absolute Error)
#   Measures the average absolute difference between predicted and true
#   signal values. Lower MAE indicates that predictions are, on average,
#   closer to the true targets.
#
# RMSE (Root Mean Squared Error)
#   Measures the square root of the average squared prediction error.
#   RMSE penalizes larger errors more heavily than MAE, making it useful
#   for detecting cases where the model produces large prediction mistakes.
#
# R² (Coefficient of Determination)
#   Indicates how much of the variance in the target signals is explained
#   by the model. Values closer to 1.0 indicate better fit, while values
#   near 0 indicate that the model performs similarly to predicting the
#   mean of the targets.

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print("\n====================")
print("Validation Set – Regression Metrics (Continuous Signal Prediction)")
print("====================")

print(
    f"MAE  (Mean Absolute Error): {val_mae:.4f}\n"
    "  Measures the average absolute difference between predicted and true\n"
    "  signal values across the validation dataset.\n"
    "  Example:\n"
    "      True signal = 0.70, Predicted = 0.65 → Error = 0.05\n"
    "      True signal = 0.30, Predicted = 0.10 → Error = 0.20\n"
    "  MAE averages these absolute errors.\n"
    "  Interpretation:\n"
    "      Lower MAE (e.g., 0.05) → predictions are numerically very close\n"
    "                              to the true signal values.\n"
    "      Higher MAE (e.g., 0.25) → predictions frequently deviate from\n"
    "                               the true values.\n"
)

print(
    f"RMSE (Root Mean Squared Error): {val_rmse:.4f}\n"
    "  Measures the square root of the average squared prediction error.\n"
    "  Squaring the errors causes larger mistakes to contribute more heavily\n"
    "  to the final score.\n"
    "  Example:\n"
    "      True signal = 0.80, Predicted = 0.60 → Error = 0.20\n"
    "      Squared error = 0.04\n"
    "  Because large errors are penalized more strongly, RMSE highlights\n"
    "  whether the model occasionally makes large prediction mistakes.\n"
    "  Interpretation:\n"
    "      Lower RMSE → predictions are consistently close to true values.\n"
    "      Higher RMSE → the model produces larger or inconsistent errors.\n"
)

print(
    f"R^2  (Coefficient of Determination): {val_r2:.4f}\n"
    "  Measures how much of the variance in the true signal values is\n"
    "  explained by the model's predictions.\n"
    "  Interpretation:\n"
    "      R^2 ≈ 1.0 → the model explains most of the variation in the data.\n"
    "      R^2 ≈ 0.0 → the model performs about the same as predicting the\n"
    "                 average signal value for every example.\n"
    "      R^2 < 0.0 → the model performs worse than predicting the mean.\n"
    "  Example:\n"
    "      If true signal values vary widely across samples and the model\n"
    "      predictions track that variation closely, R^2 will be high.\n"
)
# -----------------------------
# 6) Convert continuous signals into discrete intensity bins
# -----------------------------
# The model predicts continuous emotional signal values in the range [0, 1].
# However, emotional signals do not necessarily require perfect numerical
# precision to be useful. In many cases, it is sufficient to determine the
# approximate intensity level of a signal rather than its exact value.
#
# For example, predictions of 0.62 and 0.68 both represent a similar
# "medium" signal intensity even though the exact values differ slightly.
#
# To evaluate this type of coarse correctness, we convert continuous signal
# values into discrete intensity bins. This allows the predictions to be
# evaluated using classification metrics such as Accuracy, Precision,
# Recall, and F1.
#
# Signal intensity bins:
#
#   none      : 0.00 - 0.05
#   very_low  : 0.05 - 0.15
#   low       : 0.15 - 0.45
#   medium    : 0.45 - 0.70
#   high      : 0.70 - 1.00
#
# The bins use half-open intervals:
#
#   [0.00, 0.05)
#   [0.05, 0.15)
#   [0.15, 0.45)
#   [0.45, 0.70)
#   [0.70, 1.00]
#
# Each bin is mapped to an integer class label:
#
#   0 = none
#   1 = very_low
#   2 = low
#   3 = medium
#   4 = high
#
# This binning allows us to measure whether the model predicts the correct
# intensity level of each signal, even if the exact continuous value is not
# perfectly accurate.
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]

def bin_signal_values(arr: np.ndarray) -> np.ndarray:
    """
    Convert continuous values in [0, 1] to discrete bin IDs.
    """
    arr = np.clip(arr, 0.0, 1.0)

    binned = np.zeros_like(arr, dtype=np.int32)

    binned[(arr >= 0.05) & (arr < 0.15)] = 1
    binned[(arr >= 0.15) & (arr < 0.45)] = 2
    binned[(arr >= 0.45) & (arr < 0.70)] = 3
    binned[(arr >= 0.70)] = 4

    return binned


# Bin true and predicted values
y_val_bins_true = bin_signal_values(y_val)
y_val_bins_pred = bin_signal_values(y_val_pred_clipped)

# -----------------------------
# 7) Classification-style evaluation on validation set
# -----------------------------
# After binning continuous signal values into discrete intensity categories,
# we evaluate prediction quality using standard classification metrics.
#
# Each signal prediction is treated as a classification task where the model
# must predict the correct intensity bin (none, very_low, low, medium, high).
#
# Because each validation sample contains multiple signals, we flatten the
# signal matrices so that every signal prediction across all samples becomes
# a separate classification instance.

y_val_bins_true_flat = y_val_bins_true.reshape(-1)
y_val_bins_pred_flat = y_val_bins_pred.reshape(-1)


# Accuracy
# --------
# Measures the overall proportion of predictions where the predicted bin
# exactly matches the true bin. This gives a simple measure of how often
# the model correctly identifies the signal intensity category.

val_acc = accuracy_score(y_val_bins_true_flat, y_val_bins_pred_flat)


# Macro Precision
# ---------------
# Precision measures how often predictions of a given class are correct.
# Macro averaging computes precision independently for each class and then
# averages them equally. This prevents large classes from dominating the
# metric and ensures that rare intensity levels are evaluated fairly.

val_precision_macro = precision_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)


# Macro Recall
# ------------
# Recall measures how well the model detects all instances of each class.
# Macro averaging gives equal weight to every class, allowing us to evaluate
# whether the model successfully identifies both common and rare intensity
# levels.

val_recall_macro = recall_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)


# Macro F1 Score
# --------------
# F1 combines precision and recall into a single metric using their harmonic
# mean. The macro F1 score gives equal importance to all classes and provides
# a balanced measure of overall classification performance across intensity
# levels.

val_f1_macro = f1_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="macro",
    zero_division=0
)


# Weighted Precision
# ------------------
# Weighted precision accounts for the number of occurrences of each class
# in the dataset. Classes with more samples contribute more to the final
# score, making this metric reflect the actual class distribution.

val_precision_weighted = precision_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="weighted",
    zero_division=0
)


# Weighted Recall
# ---------------
# Similar to macro recall, but weighted by the frequency of each class.
# This gives a recall measure that reflects how well the model performs
# on the overall dataset distribution.

val_recall_weighted = recall_score(
    y_val_bins_true_flat,
    y_val_bins_pred_flat,
    average="weighted",
    zero_division=0
)

# Weighted F1 Score
# -----------------
# The weighted F1 score combines precision and recall while accounting
# for class imbalance. Classes with more samples have greater influence
# on the final metric.

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
    "  Represents the proportion of signal predictions where the predicted\n"
    "  intensity bin exactly matches the true bin.\n"
    "  Example:\n"
    "      True = medium, Predicted = medium → correct\n"
    "      True = high,   Predicted = low    → incorrect\n"
    "  A high accuracy (e.g., 0.80) means the model correctly predicts\n"
    "  the signal intensity category most of the time. A low accuracy\n"
    "  (e.g., 0.40) means the model frequently assigns the wrong bin.\n"
)

print(
    f"Precision (Macro): {val_precision_macro:.4f}\n"
    "  Precision measures how often the model's predictions for each\n"
    "  intensity class are correct. Macro precision treats all bins\n"
    "  equally regardless of how common they are.\n"
    "  Example:\n"
    "      Model predicts 'high' 100 times\n"
    "      Only 70 are actually 'high'\n"
    "      Precision = 0.70\n"
    "  A high macro precision means when the model predicts a bin, it\n"
    "  is usually correct. A low value means many predictions for a bin\n"
    "  are incorrect.\n"
)

print(
    f"Recall (Macro): {val_recall_macro:.4f}\n"
    "  Recall measures how well the model detects all true instances\n"
    "  of each intensity bin. Macro recall treats all bins equally.\n"
    "  Example:\n"
    "      There are 100 true 'high' signals\n"
    "      Model correctly identifies 65 of them\n"
    "      Recall = 0.65\n"
    "  High recall means the model successfully detects most signals\n"
    "  of that class. Low recall means the model frequently misses them.\n"
)

print(
    f"F1 Score (Macro): {val_f1_macro:.4f}\n"
    "  The F1 score combines precision and recall into a single value.\n"
    "  It is useful when you want a balance between predicting correctly\n"
    "  and detecting all true cases.\n"
    "  Example:\n"
    "      Precision = 0.80\n"
    "      Recall    = 0.60\n"
    "      F1 ≈ 0.69\n"
    "  A high F1 score indicates the model is both accurate in its\n"
    "  predictions and effective at detecting the correct bins.\n"
)

print(
    f"Precision (Weighted): {val_precision_weighted:.4f}\n"
    "  Weighted precision calculates precision for each class but\n"
    "  weights the score according to how frequently each class appears\n"
    "  in the validation dataset.\n"
    "  Example:\n"
    "      If 'low' signals occur much more frequently than 'high',\n"
    "      performance on 'low' contributes more to the final score.\n"
    "  High values indicate the model predicts the common bins reliably.\n"
)

print(
    f"Recall (Weighted): {val_recall_weighted:.4f}\n"
    "  Weighted recall measures how well the model detects true signals\n"
    "  while accounting for class frequency.\n"
    "  Example:\n"
    "      If most signals are 'low', correctly identifying those will\n"
    "      strongly increase this score.\n"
)

print(
    f"F1 Score (Weighted): {val_f1_weighted:.4f}\n"
    "  The weighted F1 score summarizes overall classification quality\n"
    "  while accounting for class imbalance in the dataset.\n"
    "  A high value indicates the model performs well across the dataset\n"
    "  distribution of signal intensities.\n"
)

# -----------------------------
# 8) Per-signal evaluation on the validation set
# -----------------------------
# Previous sections computed regression and classification metrics by
# aggregating predictions across all signals. While this provides an
# overall view of model performance, it does not reveal how well the
# model predicts each individual emotional signal.
#
# In this section, we compute the same evaluation metrics separately
# for each signal dimension. This allows us to observe which signals
# the model predicts well and which are more difficult to predict.
#
# For each signal:
#   - Regression metrics are computed using the continuous predictions.
#   - Classification metrics are computed using the binned signal values.
#
# This analysis helps identify differences in predictive performance
# across emotional dimensions (e.g., threat may be easier to predict
# than controllability or novelty).

print("\n====================")
print("Per-Signal Validation Metrics (Performance for each emotional signal individually. "
      "These metrics show how accurately the model predicts each signal dimension "
      "separately on the validation set, helping identify which signals are easier "
      "or harder for the model to learn.)")
print("====================")

for i, sig in enumerate(SIGNALS):
    # Regression metrics for this signal
    sig_mae = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
    sig_rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i]))
    sig_r2 = r2_score(y_val[:, i], y_val_pred[:, i])

    # Binned classification metrics for this signal
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


# 9) Classification report for flattened signal bins (validation set)
# -----------------------------
# This section prints a detailed classification report for the validation set
# after flattening all signal predictions across all samples. Flattening treats
# each signal prediction as an individual classification instance, allowing us
# to evaluate performance across all signal intensities together.
#
# The report provides per-class metrics for the five signal intensity bins:
# none, very_low, low, medium, high. For each class it shows precision, recall,
# F1-score, and support.

print("\n====================")
print(
    "Validation Set – Detailed Classification Report\n"
    "This report summarizes model performance for each signal intensity bin\n"
    "(none, very_low, low, medium, high) across the validation dataset.\n"
    "Each signal prediction is treated as a classification instance after\n"
    "flattening all signals across all samples.\n"
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


# -----------------------------
# 10) Final evaluation on the test set
# -----------------------------
# After model development and validation are complete, the trained model is
# evaluated on the held-out test dataset. The test set is not used during
# training or model tuning, so it provides an unbiased estimate of how well
# the model generalizes to unseen data.
#
# The evaluation follows the same structure used for the validation set:
#   1) Generate predictions for the test texts.
#   2) Compute regression metrics using the continuous signal predictions.
#   3) Convert signals into intensity bins and compute classification metrics.

# Generate predictions for the test texts
y_test_pred = baseline_model.predict(X_test_text)

# Ensure predictions stay within the valid signal range [0,1]
y_test_pred_clipped = np.clip(y_test_pred, 0.0, 1.0)


# -----------------------------
# Regression evaluation (continuous signals)
# -----------------------------
# These metrics measure how closely the predicted continuous signal values
# match the true signal values on the test set.

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)


# -----------------------------
# Classification-style evaluation (binned signals)
# -----------------------------
# Continuous signal values are converted into intensity bins
# (none, very_low, low, medium, high) so that the model can also be evaluated
# using classification metrics.

y_test_bins_true = bin_signal_values(y_test)
y_test_bins_pred = bin_signal_values(y_test_pred_clipped)

# Flatten signals so that each signal prediction is treated as an
# independent classification instance.
y_test_bins_true_flat = y_test_bins_true.reshape(-1)
y_test_bins_pred_flat = y_test_bins_pred.reshape(-1)

# Compute classification metrics for the binned predictions
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
    "Regression metrics measure numerical accuracy of the continuous signal\n"
    "predictions, while classification metrics measure how accurately the\n"
    "model predicts signal intensity categories after binning."
)
print("====================")

print(f"MAE                : {test_mae:.4f}")
print(f"RMSE               : {test_rmse:.4f}")
print(f"R^2                : {test_r2:.4f}")
print(f"Accuracy           : {test_acc:.4f}")
print(f"Precision (macro)  : {test_precision_macro:.4f}")
print(f"Recall (macro)     : {test_recall_macro:.4f}")
print(f"F1 (macro)         : {test_f1_macro:.4f}")
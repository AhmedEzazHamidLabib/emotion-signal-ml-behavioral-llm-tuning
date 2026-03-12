import json
import numpy as np
import joblib
from utils import load_signals

# -----------------------------
# Load signal definitions
# -----------------------------
# SIGNALS stores the ordered list of emotional signal names used by the model.
# The order must match the order of the target dimensions used during training,
# since each predicted value corresponds to one specific signal dimension.
#
# BIN_LABELS defines the discrete intensity categories used to convert
# continuous signal values in [0, 1] into interpretable bins.

SIGNALS = load_signals("emotion_prototypes.json")
BIN_LABELS = ["none", "very_low", "low", "medium", "high"]


# -----------------------------
# Binning helpers
# -----------------------------
# These helper functions convert continuous signal intensities into
# discrete bin labels so the model output is easier to interpret.

def bin_signal_value(v: float) -> int:
    """
    Convert a continuous signal value into an integer bin ID.

    Input:
        v : float
            Predicted signal intensity, expected to be in the range [0, 1].

    Output:
        int
            Bin ID corresponding to one of the following categories:
                0 -> none
                1 -> very_low
                2 -> low
                3 -> medium
                4 -> high

    Notes:
        The value is clipped to [0, 1] before binning to ensure it remains
        within the valid signal range.
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
    Convert a bin ID into its human-readable label.

    Example:
        3 -> "medium"
        4 -> "high"
    """
    return BIN_LABELS[bin_id]


# -----------------------------
# Prediction reporting
# -----------------------------
# This function runs inference on a single input text and prints a detailed
# report of the predicted emotional signal values.
#
# The report includes:
#   1) the original input text
#   2) raw and clipped continuous predictions for each signal
#   3) the corresponding discrete intensity bin for each signal
#   4) the strongest predicted signals
#   5) all signals reaching medium or high intensity
#   6) dictionary-style outputs for easy inspection or debugging

def print_prediction_report(text: str, model, signals):
    """
    Generate and print a detailed signal prediction report for one text input.

    Parameters:
        text : str
            Input text to evaluate.
        model :
            Trained scikit-learn pipeline containing the TF-IDF vectorizer
            and Ridge regression model.
        signals : list[str]
            Ordered list of signal names corresponding to the output
            dimensions of the model.
    """

    print("\n" + "=" * 80)
    print("INPUT TEXT FOR INFERENCE")
    print("=" * 80)
    print(text)

    # -----------------------------
    # Generate model predictions
    # -----------------------------
    # The model expects a list of texts, so the single input string is wrapped
    # in a list. The output is a 2D array with shape:
    #
    #   (number_of_inputs, number_of_signals)
    #
    # Since only one text is provided here, we take the first row [0] to get
    # the predicted signal vector for this specific input text.

    raw_pred = model.predict([text])[0]

    # Ridge regression is an unconstrained linear model, so some predicted
    # values may fall below 0 or above 1. For interpretation purposes,
    # predictions are clipped into the valid signal range [0, 1].
    clipped_pred = np.clip(raw_pred, 0.0, 1.0)

    # -----------------------------
    # Build a structured per-signal report
    # -----------------------------
    # Each row stores:
    #   - signal name
    #   - raw model prediction
    #   - clipped prediction
    #   - bin ID
    #   - bin label
    #
    # This makes it easier to reuse the same information for multiple report
    # sections below.

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

    # -----------------------------
    # Full detailed signal table
    # -----------------------------
    # This table prints every signal with:
    #   - raw predicted value
    #   - clipped value in [0,1]
    #   - intensity bin label
    #
    # Raw predictions show the direct output of the regression model.
    # Clipped predictions show the interpretable version constrained to
    # the intended signal range.

    print("\n" + "=" * 80)
    print("DETAILED SIGNAL PREDICTIONS (RAW, CLIPPED, AND BINNED)")
    print("=" * 80)
    print(f"{'Signal':20s} {'Raw':>10s} {'Clipped':>10s} {'Bin':>10s}")
    print("-" * 80)
    for row in rows:
        print(f"{row['signal']:20s} {row['raw']:10.4f} {row['clipped']:10.4f} {row['bin']:>10s}")

    # -----------------------------
    # Sort signals by predicted intensity
    # -----------------------------
    # Signals are sorted by clipped prediction in descending order so the
    # strongest predicted dimensions appear first.

    rows_sorted = sorted(rows, key=lambda r: r["clipped"], reverse=True)

    # -----------------------------
    # Top strongest signals
    # -----------------------------
    # This provides a quick summary of the most prominent emotional
    # dimensions predicted for the input text.

    print("\n" + "=" * 80)
    print("TOP 5 STRONGEST PREDICTED SIGNALS")
    print("=" * 80)
    for row in rows_sorted[:5]:
        print(f"{row['signal']:20s} {row['clipped']:.4f} ({row['bin']})")

    # -----------------------------
    # Salient signals
    # -----------------------------
    # Signals with bin_id >= 3 correspond to:
    #   3 -> medium
    #   4 -> high
    #
    # These are treated as the most salient or clearly expressed signals
    # in the current prediction.

    salient = [r for r in rows_sorted if r["bin_id"] >= 3]

    print("\n" + "=" * 80)
    print("SALIENT SIGNALS (MEDIUM OR HIGH INTENSITY)")
    print("=" * 80)
    if salient:
        for row in salient:
            print(f"{row['signal']:20s} {row['clipped']:.4f} ({row['bin']})")
    else:
        print("No signals reached medium or high intensity.")

    # -----------------------------
    # Compact dictionary output: continuous values
    # -----------------------------
    # This dictionary maps each signal to its clipped continuous prediction.
    # It is useful for debugging, logging, or exporting values for later use.

    print("\n" + "=" * 80)
    print("SIGNAL VALUE DICTIONARY (CLIPPED CONTINUOUS VALUES)")
    print("=" * 80)
    signal_dict = {row["signal"]: round(row["clipped"], 4) for row in rows}
    print(json.dumps(signal_dict, indent=2))

    # -----------------------------
    # Compact dictionary output: bin labels
    # -----------------------------
    # This dictionary maps each signal to its discrete intensity category,
    # making the output easier to interpret at a glance.

    print("\n" + "=" * 80)
    print("SIGNAL BIN DICTIONARY (DISCRETE INTENSITY LABELS)")
    print("=" * 80)
    bin_dict = {row["signal"]: row["bin"] for row in rows}
    print(json.dumps(bin_dict, indent=2))


# -----------------------------
# Main execution
# -----------------------------
# This block runs only when the script is executed directly.
#
# It:
#   1) loads the trained baseline model from disk
#   2) defines a test input string
#   3) prints a full prediction report for that input

if __name__ == "__main__":
    model = joblib.load("models/baseline_tfidf_ridge.pkl")

    # Example input text for manual testing.
    # Replace this string with any text you want to analyze.
    test_text = "I AM SO ANGRY AT YOU THAT I CANT BELIEVE WHAT TO DO"

    print_prediction_report(test_text, model, SIGNALS)
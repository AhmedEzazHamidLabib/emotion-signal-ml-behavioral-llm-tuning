from datasets import load_dataset
import json
import math
from collections import Counter
from utils import load_prototypes

# =============================================================================
# Script Purpose
# =============================================================================
# This script converts the GoEmotions dataset from discrete emotion labels
# into continuous emotional signal vectors that can be used to train a
# machine learning model.
#
# The original GoEmotions dataset contains Reddit comments annotated with
# one or more emotion labels such as:
#     joy, anger, sadness, curiosity, surprise, etc.
#
# Example original dataset entry:
#
#     {
#         "text": "This is funny but also wholesome",
#         "labels": [amusement, joy]
#     }
#
# However, emotions in real psychological systems are not isolated categories.
# Emotional experiences are better modeled as combinations of underlying
# dimensions such as:
#
#     valence        (positive vs negative experience)
#     arousal        (calm vs excited activation)
#     energy         (strength or intensity of feeling)
#     agency         (sense of control)
#     threat         (perceived danger)
#     loss           (sense of loss or grief)
#     injustice      (perceived unfairness)
#     curiosity      (desire to explore or learn)
#
# Because a single comment may contain multiple emotions simultaneously,
# we convert the discrete emotion labels into a structured emotional signal
# vector. Each signal value ranges between 0 and 1 and represents the
# intensity of that emotional dimension.
#
# Example transformed entry:
#
#     text: "This is funny but also wholesome"
#
#     signal vector:
#         valence: 0.86
#         arousal: 0.64
#         energy: 0.55
#         threat: 0.00
#         ...
#
# These signal vectors will later serve as the ground truth targets for a
# machine learning model that learns to predict emotional signals directly
# from text.
#
# The pipeline implemented in this script:
#
#     1) Load the GoEmotions dataset
#     2) Load emotion signal prototypes
#     3) Compute IDF weights for emotions
#     4) Convert emotion labels into signal vectors
#     5) Annotate each dataset example with its signal representation
#     6) Save the annotated dataset for model training




# We load the GoEmotions dataset using HuggingFace's datasets library.
# The dataset contains Reddit comments labeled with one or more emotions.
#
# Each entry in the dataset has the form:
#
#     {
#         "text": "... comment text ...",
#         "labels": [emotion_id_1, emotion_id_2, ...]
#     }
#
# The labels are stored as integer IDs which correspond to emotion names.
# We retrieve the list of emotion names so we can decode the integer labels
# into human-readable emotions when inspecting the dataset.
#
# We also print dataset statistics (number of samples in train, validation,
# and test splits) to verify that the dataset loaded correctly.
#
# These dataset splits will later be annotated with emotional signal vectors
# and used to train and evaluate our machine learning models.
# =============================================================================
# 0) Load dataset
# =============================================================================

dataset = load_dataset("go_emotions")

print(dataset["train"].features["labels"].feature.names)

sample = dataset["train"][2]
print(sample)

labels = dataset["train"].features["labels"].feature.names
decoded = [labels[i] for i in sample["labels"]]
print(decoded)

print(len(dataset["train"]))
print(len(dataset["validation"]))
print(len(dataset["test"]))
SIGNALS, proto = load_prototypes("emotion_prototypes.json")

# We count how many times each emotion label appears in the training dataset
# and compute an IDF weight using:
#       IDF = log(total_number_of_samples / count_of_samples_with_that_emotion)
#
# This is needed because a single comment can have multiple emotion labels.
# Example:
#     "This is funny and wholesome"
#     labels = [amusement, joy]
#
# If we later combine the signal vectors of multiple emotions, common emotions
# like "neutral" or "joy" might dominate simply because they appear often.
# IDF corrects this by giving rarer emotions more influence.
#
# Example:
# N = 43000 comments
#
# joy appears in 8000 comments:
#     idf[joy] = log(43000 / 8000)
#
# grief appears in 300 comments:
#     idf[grief] = log(43000 / 300)
#
# Since grief is rarer, it gets a larger IDF value and will contribute more
# when multiple emotion labels are merged into one signal vector.
#
# Example result:
# idf = {
#     0: 2.14,
#     1: 1.75,
#     2: 3.91,
#     ...
# }
#
# -----------------------------
# 1) Compute IDF on training set
# -----------------------------
def compute_idf(train_split, label_field="labels", num_labels=28):
    """
    train_split: HF Dataset split (dataset['train'])
    returns: dict[label:int] -> idf:float
    """
    label_counts = Counter()
    N = len(train_split)

    for row in train_split:
        # treat each label once per sample just to be safe
        for lab in set(row[label_field]):
            label_counts[lab] += 1

    # +1 smoothing avoids div-by-zero
    idf = {lab: math.log(N / (cnt + 1.0)) for lab, cnt in label_counts.items()}

    # Ensure all labels exist
    for lab in range(num_labels):
        idf.setdefault(lab, 0.0)

    return idf


idf = compute_idf(dataset["train"])

# We convert the IDF scores of the emotion labels into normalized weights using softmax.
# This is needed because a comment can have multiple emotion labels and we want to decide
# how much each emotion should contribute when we combine their signal vectors.
#
# Example comment:
#     "This is funny and wholesome"
#     labels = [amusement, joy]
#
# Suppose the IDF scores are:
#     idf[amusement] = 1.8
#     idf[joy] = 1.2
#
# Softmax converts these scores into weights that sum to 1:
#     weights ≈ [0.62, 0.38]
#
# This means "amusement" contributes slightly more than "joy" when we later merge
# the emotion signal vectors.
#
# These weights will later be used like:
#     final_vector =
#         w1 * proto[label1] +
#         w2 * proto[label2] +
#         ...
#
# Example:
#     final_vector =
#         0.62 * proto[amusement] +
#         0.38 * proto[joy]
#
# The temperature parameter controls how sharp the weighting is:
#     lower temperature → one emotion dominates more
#     higher temperature → weights become more even
#
# -----------------------------
# 2) Softmax weights
# -----------------------------
def softmax_weights(labels, idf, temperature=1.0, eps=1e-12):
    """
    labels: list[int]
    returns: list[float] same length, sums to 1
    """
    if len(labels) == 1:
        return [1.0]

    scores = [idf.get(lab, 0.0) / max(temperature, eps) for lab in labels]

    # stable softmax
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps) + eps
    return [e / Z for e in exps]


# We merge the signal vectors of multiple emotion labels into a single signal vector.
# This is necessary because a single comment in the GoEmotions dataset can have
# multiple emotion labels.
#
# Example comment:
#     "This is funny but also heartwarming"
#     labels = [amusement, joy]
#
# Each emotion has a prototype signal vector stored in proto:
#     proto[amusement] = {"valence": 0.80, "arousal": 0.70, "energy": 0.60, ...}
#     proto[joy]       = {"valence": 0.90, "arousal": 0.60, "energy": 0.50, ...}
#
# If multiple emotions exist, we compute softmax weights based on the IDF values
# of the emotions so that rarer emotions contribute more strongly.
#
# Example:
#     labels = [amusement, joy]
#     idf[amusement] = 1.8
#     idf[joy] = 1.2
#
# After softmax:
#     weights = [0.62, 0.38]
#
# We then combine the signal vectors using these weights.
#
# Example (for valence):
#     amusement valence = 0.80
#     joy valence = 0.90
#
#     merged valence =
#         0.62 * 0.80 + 0.38 * 0.90
#
# Some signals represent strong negative states (for example threat, loss,
# injustice, disgust). These signals are listed in ALARM.
#
# For these signals we do NOT average them because averaging would dilute
# important negative signals. Instead we take the maximum value.
#
# Example:
#     threat values = [0.1, 0.8]
#
#     weighted average → 0.45   (danger gets diluted)
#     max value        → 0.8    (danger preserved)
#
# The result is a single merged signal vector that represents the overall
# emotional signal state of the comment.
#
# Example output:
#     {
#         "valence": 0.84,
#         "arousal": 0.65,
#         "energy": 0.58,
#         "threat": 0.00,
#         ...
#     }
#
# This merged vector becomes the ground truth signal vector that the model
# will learn to predict from text.
#
# -----------------------------
# 3) Merge function
# -----------------------------
def merge_signals(labels, proto, idf, SIGNALS, ALARM, temperature=1.0):
    """
    labels: list[int]
    proto: dict[int] -> dict[str] -> float
    returns: dict[str] -> float
    """
    if len(labels) == 0:
        raise ValueError("No labels for this sample.")

    if len(labels) == 1:
        return dict(proto[labels[0]])  # copy

    w = softmax_weights(labels, idf, temperature=temperature)
    out = {}

    for sig in SIGNALS:
        vals = [proto[lab][sig] for lab in labels]

        if sig in ALARM:
            out[sig] = max(vals)
        else:
            out[sig] = sum(wi * vi for wi, vi in zip(w, vals))

    return out


# We annotate a dataset split by converting the emotion labels of each comment
# into a continuous signal vector using the merge_signals function.
#
# The GoEmotions dataset originally contains:
#     text + emotion labels
#
# Example row:
#     {
#         "text": "This is funny but also wholesome",
#         "labels": [amusement, joy]
#     }
#
# Using the emotion prototypes and the merge function, we convert the labels
# into a signal vector representing the emotional signal state of the comment.
#
# Example merged signals:
#     merged = {
#         "valence": 0.86,
#         "arousal": 0.64,
#         "energy": 0.55,
#         ...
#     }
#
# We then store this signal vector in a new column called "y".
#
# Example annotated row:
#     {
#         "text": "This is funny but also wholesome",
#         "labels": [amusement, joy],
#         "y": [0.86, 0.64, 0.55, ...]   # length = len(SIGNALS)
#     }
#
# The values in "y" follow the same order as SIGNALS:
#     SIGNALS = ["valence", "arousal", "energy", ...]
#
# This vector becomes the training target that the machine learning model
# will learn to predict from the input text.
#
# -----------------------------
# 4) Annotate a split (add "y" column)
# -----------------------------
def annotate_split(split, proto, idf, SIGNALS, ALARM, temperature=1.0, label_field="labels"):
    """
    Returns HF Dataset with new column 'y' = list[float] length len(SIGNALS)
    """
    def _annotate(example):
        labs = example[label_field]
        merged = merge_signals(
            labs,
            proto=proto,
            idf=idf,
            SIGNALS=SIGNALS,
            ALARM=ALARM,
            temperature=temperature
        )
        example["y"] = [merged[s] for s in SIGNALS]
        return example

    return split.map(_annotate, desc="Annotating split")


# -----------------------------
# 5) Define ALARM signals
# -----------------------------
# These are the dimensions where "one strong label should dominate"
# rather than being averaged away.
ALARM = {
    "threat",
    "loss",
    "injustice",
    "disgust",
    # add more only if your schema really needs it
}


# -----------------------------
# 6) Annotate all dataset splits
# -----------------------------
# For each split (train / validation / test), we convert the emotion labels
# into continuous signal vectors using the merge_signals pipeline.
#
# Each example originally looks like:
#     {
#         "text": "...",
#         "labels": [17, 1]
#     }
#
# After annotation we add a new column "y":
#     {
#         "text": "...",
#         "labels": [17, 1],
#         "y": [valence, arousal, energy, ...]
#     }
#
# The vector "y" has length len(SIGNALS) and represents the merged emotional
# signal state of the comment.
TEMPERATURE = 1.0

annotated_dataset = {
    "train": annotate_split(dataset["train"], proto, idf, SIGNALS, ALARM, temperature=TEMPERATURE),
    "validation": annotate_split(dataset["validation"], proto, idf, SIGNALS, ALARM, temperature=TEMPERATURE),
    "test": annotate_split(dataset["test"], proto, idf, SIGNALS, ALARM, temperature=TEMPERATURE),
}



# -----------------------------
# Sanity check
# -----------------------------
# Inspect one annotated example to make sure the pipeline worked correctly.
# We print the text, its emotion labels, and the resulting merged signal vector.
print("\n--- SANITY CHECK ---")
example_idx = 2
ex = annotated_dataset["train"][example_idx]

print("text:", ex["text"])
print("label ids:", ex["labels"])
print("decoded labels:", [labels[i] for i in ex["labels"]])
print("y length:", len(ex["y"]))

print("\nMerged signals:")
for s, v in zip(SIGNALS, ex["y"]):
    print(f"{s:20s} {v:.4f}")

# -----------------------------
# Save annotated dataset to disk
# -----------------------------
# We save each split to the data/ directory so we can reload it later without
# recomputing the entire annotation pipeline.
annotated_dataset["train"].save_to_disk("data/go_emotions_annotated_train")
annotated_dataset["validation"].save_to_disk("data/go_emotions_annotated_validation")
annotated_dataset["test"].save_to_disk("data/go_emotions_annotated_test")

print("\nSaved annotated splits to data/ directory.")

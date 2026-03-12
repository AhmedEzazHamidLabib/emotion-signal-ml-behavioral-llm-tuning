import json
# We load the emotion_prototypes.json file and build:
# 1) SIGNALS: an ordered list of emotion signals
#    Example:
#    SIGNALS = ["valence", "arousal", "energy", "agency", ...]
#
# 2) proto: a lookup table that maps each emotion label to its signal values
#    Example:
#    proto[17] = {
#        "valence": 0.85,
#        "arousal": 0.55,
#        "energy": 0.50,
#        ...
#    }
#
# This lets us quickly retrieve the signal vector for any emotion label.
# Example usage later:
#    proto[17]["valence"]  -> 0.85
#    proto[4]["arousal"]   -> 0.72
def load_signals(path="emotion_prototypes.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    signals_items = list(data["signals"].items())
    signals_items.sort(key=lambda kv: kv[1]["index"])
    return [name for name, meta in signals_items]


def load_prototypes(path="emotion_prototypes.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    signals_items = list(data["signals"].items())
    signals_items.sort(key=lambda kv: kv[1]["index"])
    SIGNALS = [name for name, meta in signals_items]

    proto = {}
    for row in data["emotion_prototypes"]:
        lab = int(row["label"])
        proto[lab] = {sig: float(row[sig]) for sig in SIGNALS}

    return SIGNALS, proto
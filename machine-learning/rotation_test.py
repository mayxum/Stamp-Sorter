# Tests how much stamp orientation hurts, and two fixes. Changes nothing in the sorter:
# probe_model.npz, classes.py and the Pi files are untouched.
#
# On the trapdoor a stamp can land upright, sideways or upside down, but the probe was
# trained only on upright crops. Here every labeled crop is scored at 0, 90, 180 and
# 270 degrees, as if it had been placed that way.
#
#   A. current probe                      (what the sorter does now)
#   B. current probe, best of 4 rotations (sorter tries all 4 and keeps the most confident)
#   C. probe retrained on all 4 rotations (same speed on the Pi, just a new probe file)
#   D. C, averaged over the 4 rotations   (4x the CLIP work per stamp on the Pi)
#
# Same evaluation as probe.py: 5-fold StratifiedGroupKFold (seed 0), C=10, balanced.
# Also scores the Pi crops in live_crops/ (labeled ones in live_crops/<class>/).
#
# Run on the Mac, in the code folder, with the venv active:
#     python rotation_test.py
# The first run encodes every crop 4 times (roughly 10 minutes) and caches the result
# in rotation_cache.npz, so later runs take seconds. Results go to rotation_results.txt.
# It also writes probe_model_rot.npz (method C) for later; the sorter does not use it.

import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from transformers import CLIPModel, CLIPProcessor

from classes import CLASSES, MODEL

C = 10.0
FOLDS = 5
THRESHOLD = 0.70
ROTS = (0, 90, 180, 270)
CACHE = "rotation_cache.npz"


def load_labeled(root="data/upright"):
    paths, labels, groups = [], [], []
    for index, name in enumerate(CLASSES):
        for path in sorted(glob.glob(os.path.join(root, name, "*.png"))):
            paths.append(path)
            labels.append(index)
            stem = os.path.splitext(os.path.basename(path))[0]
            groups.append("%s/%s" % (name, stem.rsplit("_", 1)[0]))
    return paths, np.array(labels), groups


def load_live():
    paths, labels = [], []
    for index, name in enumerate(CLASSES):
        for path in sorted(glob.glob(os.path.join("live_crops", name, "*.png"))):
            paths.append(path)
            labels.append(index)
    unlabeled = sorted(glob.glob(os.path.join("live_crops", "*.png")))
    return paths, np.array(labels, dtype=int), unlabeled


def embed(paths, model, processor):
    # Returns an array (4 rotations, N, 768): same embedding as the sorter, normalised.
    embeddings = np.zeros((len(ROTS), len(paths), 768), dtype=np.float32)
    done = 0
    for start in range(0, len(paths), 8):
        images = []
        for path in paths[start : start + 8]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                images += [image.rotate(rotation, expand=True) for rotation in ROTS]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            features = model.visual_projection(
                model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            )
            features = (features / features.norm(dim=-1, keepdim=True)).numpy()
        batch_size = len(images) // len(ROTS)
        embeddings[:, start : start + batch_size] = features.reshape(
            batch_size, len(ROTS), -1
        ).transpose(1, 0, 2)
        done += batch_size
        print("\r  %d/%d" % (done, len(paths)), end="", flush=True)
    print()
    return embeddings


def fit(X, y):
    classifier = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    classifier.fit(X, y)
    return classifier


def fit_rot(F, y):
    # Train on every rotation of every crop.
    return fit(np.concatenate(list(F)), np.tile(y, len(ROTS)))


def per_rotation(clf, F):
    return np.stack([clf.predict_proba(F[r]) for r in range(len(ROTS))])


def best_of(P):
    # For each stamp, keep the rotation with the highest confidence. P: (4, N, classes).
    rotation_indices = P.max(axis=2).argmax(axis=0)
    return P[rotation_indices, np.arange(P.shape[1])]


def stats(P, y):
    predictions, confidence = P.argmax(1), P.max(1)
    keep = confidence >= THRESHOLD
    accepted_accuracy = (predictions[keep] == y[keep]).mean() if keep.any() else 0.0
    return (
        (predictions == y).mean(),
        accepted_accuracy,
        (predictions[keep] != y[keep]).mean() if keep.any() else 0.0,
        1 - keep.mean(),
    )


def main():
    if not os.path.isdir("data/upright"):
        sys.exit("Run this from the code folder (data/upright not found).")
    paths, training_labels, groups = load_labeled()
    unsorted = sorted(glob.glob("data/unsorted/*.png"))
    live, live_y, live_unlabeled = load_live()
    print(
        "%d labeled, %d unsorted, %d labeled Pi crops, %d unlabeled Pi crops"
        % (len(paths), len(unsorted), len(live), len(live_unlabeled))
    )

    all_paths = paths + unsorted + live + live_unlabeled
    features = None
    if os.path.exists(CACHE):
        cache_data = np.load(CACHE, allow_pickle=True)
        if list(cache_data["paths"]) == all_paths:
            features = cache_data["features"]
            print("using cached embeddings (%s)" % CACHE)
    if features is None:
        processor = CLIPProcessor.from_pretrained(MODEL)
        model = CLIPModel.from_pretrained(MODEL).eval()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        print("Encoding every crop at 0/90/180/270 degrees...")
        features = embed(all_paths, model, processor)
        np.savez(CACHE, features=features, paths=np.array(all_paths))
    training_end, unsorted_end, live_end = (
        len(paths),
        len(paths) + len(unsorted),
        len(paths) + len(unsorted) + len(live),
    )
    training_features, unsorted_features, live_features, unlabeled_features = (
        features[:, :training_end],
        features[:, training_end:unsorted_end],
        features[:, unsorted_end:live_end],
        features[:, live_end:],
    )

    # Cross-validated probabilities for every rotation
    split = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    current_probabilities = np.zeros((len(ROTS), len(training_labels), len(CLASSES)))
    retrained_probabilities = np.zeros_like(current_probabilities)
    for train_indices, test_indices in split.split(training_features[0], training_labels, groups):
        current_probabilities[:, test_indices] = per_rotation(
            fit(training_features[0][train_indices], training_labels[train_indices]),
            training_features[:, test_indices],
        )
        retrained_probabilities[:, test_indices] = per_rotation(
            fit_rot(training_features[:, train_indices], training_labels[train_indices]),
            training_features[:, test_indices],
        )
    current_probe, retrained_probe = fit(training_features[0], training_labels), fit_rot(
        training_features, training_labels
    )
    unsorted_current, unsorted_retrained = per_rotation(
        current_probe, unsorted_features
    ), per_rotation(retrained_probe, unsorted_features)

    def placed(P):  # one row per (stamp, way it was placed)
        return P.reshape(-1, P.shape[2]), np.tile(training_labels, len(ROTS))

    rows = [
        (
            "A. current probe, stamps upright",
            current_probabilities[0],
            training_labels,
            unsorted_current[0],
        ),
        (
            "A. current probe, any rotation",
            *placed(current_probabilities),
            unsorted_current.reshape(-1, len(CLASSES)),
        ),
        (
            "B. current probe, best of 4",
            best_of(current_probabilities),
            training_labels,
            best_of(unsorted_current),
        ),
        (
            "C. retrained on 4 rotations",
            *placed(retrained_probabilities),
            unsorted_retrained.reshape(-1, len(CLASSES)),
        ),
        (
            "D. C averaged over 4 rotations",
            retrained_probabilities.mean(0),
            training_labels,
            unsorted_retrained.mean(0),
        ),
    ]
    lines = [
        "%-36s %6s | at %.2f: %6s %8s %6s %10s"
        % ("method", "acc", THRESHOLD, "acc", "misfile", "rej", "unsorted")
    ]
    for name, P, labels, Pu in rows:
        accuracy, accepted_accuracy, wrong, rejected_fraction = stats(P, labels)
        unsorted_rejected = (Pu.max(1) < THRESHOLD).mean()
        lines.append(
            "%-36s %5.1f%% | %15.1f%% %7.1f%% %5.0f%% %9.0f%%"
            % (
                name,
                100 * accuracy,
                100 * accepted_accuracy,
                100 * wrong,
                100 * rejected_fraction,
                100 * unsorted_rejected,
            )
        )

    # Pi crops, scored as they were photographed (models trained on data/upright only)
    def live_methods(Z):
        current, retrained = per_rotation(current_probe, Z), per_rotation(retrained_probe, Z)
        return {"A": current[0], "B": best_of(current), "C": retrained[0], "D": retrained.mean(0)}

    if len(live):
        lines.append(
            "\nLabeled Pi crops (%d), as photographed: right / rejected / wrong bin" % len(live)
        )
        for key, P in live_methods(live_features).items():
            predictions, confidence = P.argmax(1), P.max(1)
            keep = confidence >= THRESHOLD
            lines.append(
                "  %s: %3d right  %3d rejected  %3d wrong"
                % (
                    key,
                    (keep & (predictions == live_y)).sum(),
                    (~keep).sum(),
                    (keep & (predictions != live_y)).sum(),
                )
            )
    if len(live_unlabeled):
        lines.append(
            "\nUnlabeled Pi crops (%d): how many each method accepts (not whether right)"
            % len(live_unlabeled)
        )
        lines.append("%-34s %s" % ("crop", "   ".join("%-12s" % k for k in "ABCD")))
        methods = live_methods(unlabeled_features)
        for i, path in enumerate(live_unlabeled):
            cells = []
            for P in methods.values():
                best = int(P[i].argmax())
                tag = CLASSES[best][:3] if P[i, best] >= THRESHOLD else "rej"
                cells.append("%s %3.0f%%" % (tag, 100 * P[i, best]))
            lines.append(
                "%-34s %s"
                % (
                    os.path.basename(path)[:34],
                    "   ".join("%-12s" % live_end for live_end in cells),
                )
            )

    lines.append(
        "\nacc = cross-validated accuracy. At the threshold: acc = accuracy on accepted stamps,\n"
        "misfile = accepted stamps sent to the wrong bin, rej = labeled stamps sent to reject,\n"
        "unsorted = unsorted pile sent to reject (higher is better).\n"
        "'any rotation' rows count every crop 4 times, once per way it could be placed."
    )
    print("\n" + "\n".join(lines))
    with open("rotation_results.txt", "w") as results_file:
        results_file.write("\n".join(lines) + "\n")
    np.savez(
        "probe_model_rot.npz",
        coef=retrained_probe.coef_,
        intercept=retrained_probe.intercept_,
        classes=np.array(CLASSES),
        model=np.array(MODEL),
    )
    print("\nsaved rotation_results.txt and probe_model_rot.npz (not used by the sorter yet)")


if __name__ == "__main__":
    main()

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from transformers import CLIPModel, CLIPProcessor

from classes import CLASSES, PROMPTS, MODEL
from window import Window


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/upright")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cache", default="features.npz")
    parser.add_argument("--show", action="store_true",
                        help="replay the cross-validated results in a live window")
    parser.add_argument("--pace", type=float, default=0.45,
                        help="seconds per stamp when --show is on")
    parser.add_argument("--C", type=float, default=10.0,
                        help="regularisation, lower = stronger. 10 measured best; "
                             "below 1 underfits badly (0.001 gave 59%%)")
    return parser.parse_args()


def load_dataset(args):
    paths, labels, groups = [], [], []
    for index, name in enumerate(CLASSES):
        for path in sorted(glob.glob(os.path.join(args.data, name, "*.png"))):
            paths.append(path)
            labels.append(index)
            stem = os.path.splitext(os.path.basename(path))[0]
            groups.append("%s/%s" % (name, stem.rsplit("_", 1)[0]))
    if not paths:
        raise SystemExit("no crops under %s" % args.data)
    labels = np.array(labels)
    return paths, labels, groups


def load_features(args, paths, model, processor):
    cached = None
    if os.path.exists(args.cache):
        blob = np.load(args.cache, allow_pickle=True)
        if list(blob["paths"]) == paths and str(blob["model"]) == args.model:
            cached = blob["features"]
            print("using cached features (%s)" % args.cache)

    if cached is None:
        print("encoding %d crops with %s..." % (len(paths), args.model))
        chunks = []
        for start in range(0, len(paths), 32):
            batch = [Image.open(p).convert("RGB") for p in paths[start:start + 32]]
            inputs = processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            chunks.append((feats / feats.norm(dim=-1, keepdim=True)).numpy())
        cached = np.concatenate(chunks)
        np.savez(args.cache, features=cached, paths=np.array(paths),
                 model=np.array(args.model))
    return cached


def print_report(name, predictions, labels):
    print("\n%s" % name)
    print("  %-9s %5s %5s  %s" % ("class", "n", "hit", "accuracy"))
    total = hits = 0
    for index, label in enumerate(CLASSES):
        mask = labels == index
        n = int(mask.sum())
        hit = int((predictions[mask] == index).sum())
        total += n
        hits += hit
        print("  %-9s %5d %5d  %3d%%" % (label, n, hit, 100 * hit / max(1, n)))
    print("  %-9s %5d %5d  %3d%%" % ("OVERALL", total, hits, 100 * hits / total))
    print("\n  confusion (rows = truth, cols = predicted)")
    print("  %-9s" % "" + "".join("%-9s" % c for c in CLASSES))
    for index, label in enumerate(CLASSES):
        row = [int(((labels == index) & (predictions == j)).sum())
               for j in range(len(CLASSES))]
        print("  %-9s" % label + "".join("%-9s" % (v or "") for v in row))


def cross_validate(args, features, labels, groups):
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    probe = np.empty_like(labels)
    probabilities = np.zeros((len(labels), len(CLASSES)))
    for train_index, test_index in splitter.split(features, labels, groups):
        classifier = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
        classifier.fit(features[train_index], labels[train_index])
        probabilities[test_index] = classifier.predict_proba(features[test_index])
        probe[test_index] = probabilities[test_index].argmax(axis=1)
    return probe, probabilities


def main():
    args = parse_args()

    paths, labels, groups = load_dataset(args)

    model = CLIPModel.from_pretrained(args.model).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    features = load_features(args, paths, model, processor)

    # Compare zero-shot predictions on the same images.
    inputs = processor(text=[PROMPTS[c] for c in CLASSES], return_tensors="pt", padding=True)
    with torch.no_grad():
        text = model.get_text_features(**inputs)
    text = (text / text.norm(dim=-1, keepdim=True)).numpy()
    zero_shot = (features @ text.T).argmax(axis=1)

    # Keep every shot of a stamp in the same validation fold.
    probe, probabilities = cross_validate(args, features, labels, groups)

    print_report("ZERO-SHOT (prompts only, no training)", zero_shot, labels)
    print_report("LINEAR PROBE (%d-fold cross validation)" % args.folds, probe, labels)

    # Train the deployment model on all available images.
    final = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    final.fit(features, labels)
    np.savez("probe_model.npz", coef=final.coef_, intercept=final.intercept_,
             classes=np.array(CLASSES), model=np.array(args.model))
    print("\nprobe trained on all data -> probe_model.npz")
    print("that is %d numbers: a %s matrix plus %d biases."
          % (final.coef_.size + final.intercept_.size, final.coef_.shape, len(CLASSES)))

    if args.show:
        win = Window("CLIP + trained classifier",
                     "every stamp judged by a model that never saw it",
                     CLASSES, pace=args.pace)
        for index, path in enumerate(paths):
            if not win.update(path, CLASSES[labels[index]], scores=probabilities[index]):
                break
        win.close()


if __name__ == "__main__":
    main()

# Stamp classifier worker: finds the stamp in a photo, crops it, and scores it with
# CLIP ViT-L/14 + the trained linear probe (same math as pi_classify.py).
#
# Runs in the CLIP environment. sorter_app.py starts it in the background and talks
# to it one JSON line at a time:
#     request : {"photo": "/path/photo.jpg", "roi": [x, y, w, h] or null}
#     response: {"ok": true, "scores": [...4...], "crop": "/path/crop.png", ...}
#               {"ok": false, "error": "no stamp found"}
#
# Needs these files in the same folder: classes.py, sheet.py, probe_model.npz.
#
# Quick test on one photo (Mac or Pi):
#     python classifier_worker.py --test ~/Downloads/stamps2.jpg

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = sys.stdout  # JSON replies only; everything else goes to stderr

with contextlib.redirect_stdout(sys.stderr):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    import sheet
    from classes import CLASSES, MODEL, REJECT_THRESHOLD

MIN_FIT = 0.80  # ignore detections that are not stamp-shaped
# a crop this similar to the empty-plate fingerprint = no stamp
EMPTY_SIM = 0.83
# crop darker than this (0-255 average) = bare plate; plate ~60-78, stamps 143+
DARK_MAX = 100
CROP_DIR = os.path.join(HERE, "crops")


def call_find(bgr):
    # sheet.find_stamps, tolerant of how its debug argument behaves.
    try:
        result = sheet.find_stamps(bgr, debug=None)
    except Exception:
        result = sheet.find_stamps(
            bgr, debug=os.path.join(tempfile.gettempdir(), "sorter_mask.png")
        )
    return result[0] if isinstance(result, tuple) else result


def roi_region(bgr, roi):
    if not roi:
        return bgr
    x, y, w, h = roi
    return bgr[y : y + h, x : x + w]


def find_one(bgr, roi, pad=(60, 60, 60)):
    # Return (deskewed crop, fit, number found) for the stamp nearest the center.
    if roi:
        x, y, w, h = roi
        bgr = bgr[y : y + h, x : x + w]
        # sheet.py ignores blobs larger than 20% of the image. Inside a tight trapdoor
        # crop one stamp is much bigger than that, so pad the crop. The padding is the
        # plate's own colour: with black padding the dark plate itself looked like a
        # stamp, and small stamps were lost inside it.
        padding_height, padding_width = bgr.shape[0], bgr.shape[1]
        bgr = cv2.copyMakeBorder(
            bgr,
            padding_height,
            padding_height,
            padding_width,
            padding_width,
            cv2.BORDER_CONSTANT,
            value=pad,
        )
    found = [stamp for stamp in call_find(bgr) if stamp["fit"] >= MIN_FIT]
    if not found:
        return None, 0.0, 0
    height, width = bgr.shape[:2]
    areas = [stamp["rect"][1][0] * stamp["rect"][1][1] for stamp in found]
    largest_area = max(areas)
    best = min(
        (stamp for stamp, area in zip(found, areas) if area >= 0.5 * largest_area),
        key=lambda stamp: (stamp["rect"][0][0] - width / 2) ** 2
        + (stamp["rect"][0][1] - height / 2) ** 2,
    )
    return sheet.deskew(bgr, best["rect"]), float(best["fit"]), len(found)


class Classifier:
    def __init__(self):
        with np.load(os.path.join(HERE, "probe_model.npz")) as probe:
            self.weights = probe["coef"].copy()
            self.bias = probe["intercept"].copy()
            if str(probe["model"].item()) != MODEL or list(probe["classes"]) != CLASSES:
                raise ValueError("probe_model.npz does not match classes.py")
        torch.set_num_threads(4)
        self.processor = CLIPProcessor.from_pretrained(MODEL)
        self.model = CLIPModel.from_pretrained(MODEL).eval()
        os.makedirs(CROP_DIR, exist_ok=True)
        self.empty_refs = []  # CLIP fingerprints of the empty trapdoor
        self.pad = (60, 60, 60)  # plate colour (BGR), measured from the empty photo

    def embed(self, crop_bgr):
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
            features = self.model.visual_projection(vision.pooler_output)
            return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

    def scores(self, crop_bgr, feats=None):
        if feats is None:
            feats = self.embed(crop_bgr)
        logits = (feats @ self.weights.T + self.bias)[0]
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def set_empty(self, request):
        # Fingerprint the empty trapdoor: the whole box, plus whatever the detector picks out of it.
        bgr = cv2.imread(request["empty"])
        if bgr is None:
            return {"ok": False, "error": "could not read photo"}
        roi = request.get("roi")
        region = roi_region(bgr, roi)
        self.pad = tuple(float(v) for v in region.reshape(-1, 3).mean(axis=0))
        self.empty_refs = [self.embed(region)]
        crop, _, _ = find_one(bgr, roi, self.pad)
        if crop is not None:
            self.empty_refs.append(self.embed(crop))
        print("empty trapdoor fingerprint set (%d)" % len(self.empty_refs), file=sys.stderr)
        return {"ok": True, "empty_set": len(self.empty_refs)}

    def empty_similarity(self, feats):
        return max(
            (float((feats @ reference.T)[0, 0]) for reference in self.empty_refs), default=0.0
        )

    def handle(self, request):
        if "empty" in request:
            return self.set_empty(request)
        start = time.perf_counter()
        bgr = cv2.imread(request["photo"])
        if bgr is None:
            return {"ok": False, "error": "could not read photo"}
        crop, fit, found_count = find_one(bgr, request.get("roi"), self.pad)
        if crop is None:
            region = roi_region(bgr, request.get("roi"))
            roi_brightness = float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean())
            if roi_brightness < DARK_MAX:
                return {"ok": False, "error": "no stamp found", "roi_brightness": roi_brightness}
            # Something bright fills the box but has no clean outline: usually a big stamp
            # hanging over the plate edge. Classify the whole box instead of giving up.
            print("no clean outline; classifying the whole box", file=sys.stderr)
            crop, fit, found_count = region, 0.0, 0
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        if brightness < DARK_MAX:  # stamps are light paper, the plate is dark
            print(
                "crop brightness %.0f (under %d = empty plate)" % (brightness, DARK_MAX),
                file=sys.stderr,
            )
            return {"ok": False, "empty": True, "error": "empty trapdoor", "brightness": brightness}
        features = self.embed(crop)
        similarity = self.empty_similarity(features)
        print(
            "empty-trapdoor similarity %.2f (%.2f or more = empty)" % (similarity, EMPTY_SIM),
            file=sys.stderr,
        )
        if similarity >= EMPTY_SIM:
            return {"ok": False, "empty": True, "error": "empty trapdoor", "empty_sim": similarity}
        probabilities = self.scores(crop, features)
        crop_name = os.path.splitext(os.path.basename(request["photo"]))[0] + "_crop.png"
        crop_path = os.path.join(CROP_DIR, crop_name)
        cv2.imwrite(crop_path, crop)
        best = int(probabilities.argmax())
        return {
            "ok": True,
            "scores": [float(v) for v in probabilities],
            "classes": CLASSES,
            "best": best,
            "confidence": float(probabilities[best]),
            "accepted": bool(probabilities[best] >= REJECT_THRESHOLD),
            "threshold": REJECT_THRESHOLD,
            "crop": crop_path,
            "fit": fit,
            "found": found_count,
            "empty_sim": similarity,
            "seconds": time.perf_counter() - start,
        }


def reply(obj):
    OUT.write(json.dumps(obj) + "\n")
    OUT.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="classify one photo and print the result")
    parser.add_argument("--roi", help="x,y,w,h crop for --test")
    args = parser.parse_args()

    start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        classifier = Classifier()
    load_seconds = time.perf_counter() - start

    if args.test:
        roi = [int(v) for v in args.roi.split(",")] if args.roi else None
        with contextlib.redirect_stdout(sys.stderr):
            result = classifier.handle({"photo": args.test, "roi": roi})
        print("model loaded in %.1f s" % load_seconds)
        if not result["ok"]:
            print("FAILED:", result["error"])
            return
        for name, value in zip(CLASSES, result["scores"]):
            print("  %-8s %5.1f%%" % (name, 100 * value))
        verdict = CLASSES[result["best"]] if result["accepted"] else "reject"
        print(
            "-> %s (%.0f%%), fit %.3f, %d stamp(s) found, %.2f s"
            % (
                verdict,
                100 * result["confidence"],
                result["fit"],
                result["found"],
                result["seconds"],
            )
        )
        print("crop saved to", result["crop"])
        return

    reply(
        {
            "ready": True,
            "load_seconds": load_seconds,
            "classes": CLASSES,
            "threshold": REJECT_THRESHOLD,
        }
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = classifier.handle(json.loads(line))
        except Exception as error:  # keep the worker alive for the next stamp
            result = {"ok": False, "error": "%s: %s" % (type(error).__name__, error)}
        reply(result)


if __name__ == "__main__":
    main()

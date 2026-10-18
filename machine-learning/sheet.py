import argparse
import os
import sys

import cv2
import numpy as np

# Keep stamp-sized regions and ignore tiny specks or the whole background.
MIN_FRAC = 0.0015
MAX_FRAC = 0.20
ROW_TOL = 0.06  # Row spacing tolerance, as a fraction of the image height.


def find_stamps(bgr, debug=None):
    """Find stamp outlines and return their measurements and the binary mask."""
    height, width = bgr.shape[:2]
    image_area = float(height * width)

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)

    _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Treat the image border as background and make it black in the mask.
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    kernel_size = max(3, int(round(min(height, width) * 0.002)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close gaps in the perforated border before filling the stamp interior.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dark stamps may show up as hollow outlines. Fill the enclosed areas
    # by flooding the background from the corner and inverting it.
    flooded = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(flooded))

    # Remove small noise without eroding thin stamp borders.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        cv2.imwrite(debug, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stamps = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_FRAC * image_area <= area <= MAX_FRAC * image_area):
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rect_width, rect_height), _ = rect
        if rect_width < 10 or rect_height < 10:
            continue
        fit = area / (rect_width * rect_height)
        aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)
        if aspect_ratio > 3.0:  # Ignore long, thin regions.
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        clipped = (x <= 2 or y <= 2 or x + box_width >= width - 2 or y + box_height >= height - 2)
        stamps.append(dict(cnt=contour, rect=rect, fit=fit, area=area,
                         cx=cx, cy=cy, clipped=clipped))
    return stamps, mask


def row_major(items, frame_h):
    """Group stamps into rows, then sort each row from left to right."""
    items = sorted(items, key=lambda stamp: stamp["cy"])
    rows = []
    current_row = []
    for item in items:
        if current_row and abs(item["cy"] - current_row[-1]["cy"]) > ROW_TOL * frame_h:
            rows.append(current_row)
            current_row = []
        current_row.append(item)
    if current_row:
        rows.append(current_row)
    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda stamp: stamp["cx"]))
    return ordered


def deskew(bgr, rect):
    """Rotate and crop a stamp, keeping the orientation it had on the table.

    minAreaRect gives two equivalent ways to square up the same box, 90 degrees
    apart. Taking the smaller rotation leaves a portrait stamp portrait and a
    landscape one landscape, instead of forcing everything landscape. Sideways
    crops measurably cost zero-shot accuracy, upside down ones do not.
    """
    (cx, cy), (rect_width, rect_height), angle = rect
    if angle > 45.0:
        rect_width, rect_height = rect_height, rect_width
        angle -= 90.0
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(bgr, rotation_matrix, (bgr.shape[1], bgr.shape[0]),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.getRectSubPix(rotated, (int(round(rect_width)), int(round(rect_height))), (cx, cy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--cls", required=True, help="class folder name")
    parser.add_argument("--out", default="data/upright")
    parser.add_argument("--shot", default="a")
    parser.add_argument("--start", type=int, default=1, help="first stamp id")
    parser.add_argument("--dry", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    bgr = cv2.imread(args.photo)
    if bgr is None:
        sys.exit("could not read %s" % args.photo)
    height, width = bgr.shape[:2]
    print("image %d x %d" % (width, height))

    mask_path = os.path.splitext(args.photo)[0] + "_mask.png"
    found, _ = find_stamps(bgr, debug=mask_path)
    found = row_major(found, height)
    print("found %d stamps  (mask written to %s)\n" % (len(found), mask_path))

    output_dir = os.path.join(args.out, args.cls)
    if not args.dry:
        os.makedirs(output_dir, exist_ok=True)

    print("  id   fit    size(px)      notes")
    for stamp_id, stamp in enumerate(found, start=args.start):
        (rect_width, rect_height) = stamp["rect"][1]
        notes = []
        if stamp["clipped"]:
            notes.append("CUT OFF AT FRAME EDGE")
        if stamp["fit"] < 0.80:
            notes.append("fit very low, wrong object?")
        elif stamp["fit"] < 0.88:
            notes.append("loose fit, check glare")
        print("  %03d  %.3f  %4d x %4d   %s"
              % (stamp_id, stamp["fit"], int(rect_width), int(rect_height), ", ".join(notes)))
        if not args.dry:
            crop = deskew(bgr, stamp["rect"])
            cv2.imwrite(os.path.join(output_dir, "%03d_%s.png" % (stamp_id, args.shot)), crop)

    if not args.dry:
        print("\nwrote %d files to %s/" % (len(found), output_dir))


if __name__ == "__main__":
    main()

import os
import glob
import cv2
import numpy as np

# ----------------------------
# Tunable parameters
# ----------------------------
INPUT_DIR = "/home/oren/Documents/Models/Order_phase/CLIP_class/test4/Raw_data/mixed/"
OUTPUT_DIR = "/home/oren/Documents/Models/Order_phase/CLIP_class/test4/Raw_data/mixed/save/"

import cv2
import numpy as np

# --- Tunables ---
SEARCH_BOTTOM_FRACTION = 0.30     # search bottom 30%
PERCENTILE_FOR_THRESH = 98.0       # auto-threshold using this percentile in the bottom band
MIN_ASPECT_RATIO = 7.0             # wide & thin
MAX_RELATIVE_HEIGHT = 0.05         # bar height <= 5% of image height
MIN_AREA_FRACTION = 0.0003
MAX_AREA_FRACTION = 0.08
REQUIRE_TOUCH_BOTTOM = True        # bar should nearly touch the bottom edge
MAX_GAP_FROM_BOTTOM_PX = 3         # how much gap to allow if REQUIRE_TOUCH_BOTTOM
PAD_ABOVE_BAR_PX = 6               # extra pixels above the bar to crop
FALLBACK_CROP_PX = 100             # last-resort trim if detection confidence is low
MORPH_K = 5                        # morphology kernel size
DEBUG_SAVE = False                 # set True to save debug overlays
DEBUG_DIR = "_debug"
# ----------------

def _auto_bright_threshold(roi_gray, percentile=PERCENTILE_FOR_THRESH):
    # Robust per-image threshold from the bottom band
    p = np.clip(percentile, 50.0, 100.0)
    t = np.percentile(roi_gray, p)
    # keep threshold within [160, 245] to avoid extremes
    return int(np.clip(t, 160, 245))

def _rectangularity_score(w, h, area, contour):
    # Favor rectangles close to their bounding box area (solidity-like)
    rect_area = w * h
    if rect_area <= 0:
        return 0.0
    rect_score = area / rect_area  # 0..1, higher is more rectangular
    return rect_score

def _projection_scan_bottom(img_gray, search_top, min_run_frac=0.6):
    """
    Bottom-up scan: for each row from bottom to search_top, compute the fraction
    of bright pixels; return the first row that has a long bright run.
    Useful when the bar is a continuous bright strip touching the bottom.
    """
    band = img_gray[search_top:, :]
    # Otsu within the band to get a fallback binary
    _, band_bin = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # compute per-row bright fraction
    row_sum = (band_bin == 255).sum(axis=1) / float(band_bin.shape[1])
    # iterate from bottom to top
    for i in range(len(row_sum)-1, -1, -1):
        if row_sum[i] >= min_run_frac:
            # convert band index to full-image y; bar top is around here
            return search_top + i
    return None

def crop_scale_bar_bottom(img_gray):
    """
    Detect a wide bright scale bar along the bottom and crop above it.
    Falls back to a fixed-pixel bottom trim if confidence is low.
    """
    h, w = img_gray.shape[:2]
    search_top = int(h * (1.0 - SEARCH_BOTTOM_FRACTION))
    roi = img_gray[search_top:h, 0:w]

    # 1) Per-image bright threshold from bottom band
    thr_val = _auto_bright_threshold(roi)
    _, bin_bright = cv2.threshold(roi, thr_val, 255, cv2.THRESH_BINARY)

    # 2) Morphological close to merge fragmented bar pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    closed = cv2.morphologyEx(bin_bright, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) Contour-based candidates
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)
    best = None
    best_score = -1.0

    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        y_full = y + search_top
        area = cv2.contourArea(c)
        if area < MIN_AREA_FRACTION * img_area or area > MAX_AREA_FRACTION * img_area:
            continue
        aspect = cw / max(1.0, ch)
        if aspect < MIN_ASPECT_RATIO:
            continue
        if ch > MAX_RELATIVE_HEIGHT * h:
            continue

        # bottom-touch constraint
        dist_to_bottom = h - (y_full + ch)
        if REQUIRE_TOUCH_BOTTOM and dist_to_bottom > MAX_GAP_FROM_BOTTOM_PX:
            continue

        rect_score = _rectangularity_score(cw, ch, area, c)  # 0..1
        width_norm = cw / float(w)                           # 0..1

        # confidence score: prefer wide, rectangular, bottom-touching blobs
        score = (2.0 * width_norm) + (aspect / 50.0) + (2.0 * rect_score) + (1.0 - min(1.0, dist_to_bottom / 5.0))

        if score > best_score:
            best_score = score
            best = (x, y_full, cw, ch, dist_to_bottom, score)

    # 4) If a good candidate found, crop above it
    if best is not None:
        x, y_full, cw, ch, gap, score = best
        crop_to = max(0, y_full - PAD_ABOVE_BAR_PX)
        cropped = img_gray[0:crop_to, 0:w]

        if DEBUG_SAVE:
            import os
            os.makedirs(DEBUG_DIR, exist_ok=True)
            dbg = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(dbg, (x, y_full), (x+cw, y_full+ch), (0, 255, 0), 2)
            cv2.line(dbg, (0, crop_to), (w, crop_to), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, "detected_debug.png"), dbg)

        return cropped, {
            "method": "detected",
            "bar_box": [int(x), int(y_full), int(cw), int(ch)],
            "score": float(score),
            "thr_val": int(thr_val)
        }

    # 5) Try projection scan as a backup if nothing matched the heuristics
    y_proj = _projection_scan_bottom(img_gray, search_top)
    if y_proj is not None:
        crop_to = max(0, y_proj - PAD_ABOVE_BAR_PX)
        cropped = img_gray[0:crop_to, 0:w]
        if DEBUG_SAVE:
            import os
            os.makedirs(DEBUG_DIR, exist_ok=True)
            dbg = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2BGR)
            cv2.line(dbg, (0, crop_to), (w, crop_to), (0, 255, 255), 2)
            cv2.imwrite(os.path.join(DEBUG_DIR, "projection_debug.png"), dbg)
        return cropped, {"method": "projection", "y_proj": int(y_proj), "thr_val": int(thr_val)}

    # 6) Last resort: fixed bottom trim
    cropped = img_gray[0:max(0, h - FALLBACK_CROP_PX), 0:w]
    return cropped, {"method": "fallback", "removed_px": FALLBACK_CROP_PX, "thr_val": int(thr_val)}


def process_image(image_path, output_dir):
    # Load grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")

    # 1) Remove the (bottom) scale bar
    img_gray_cropped, info = crop_scale_bar_bottom(img_gray)

    # 2) Your existing pipeline (blur + adaptive threshold)
    blurred = cv2.GaussianBlur(img_gray_cropped, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 6
    )

    # Save
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_edited.png")
    cv2.imwrite(out_path, thresh)

    return out_path, info


def main():
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, p)))

    if not image_paths:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, img_path in enumerate(sorted(image_paths), 1):
        try:
            out_path, info = process_image(img_path, OUTPUT_DIR)
            print(f"[{i}/{len(image_paths)}] Saved: {out_path}  |  scale-bar removal: {info}")
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] FAILED: {img_path}  |  {e}")

if __name__ == "__main__":
    main()

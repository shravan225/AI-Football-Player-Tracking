"""
define_roi.py — Interactive Football Field ROI Polygon Tool
============================================================
Draw a polygon around the football pitch on a single video frame.
Saves the result to JSON for use by track_football.py.

Controls:
  Left-click      — add vertex
  Right-click     — undo last vertex
  Enter / Space   — save polygon
  R               — reset
  Q / Escape      — quit without saving

Usage:
    python define_roi.py --input video.mp4 --output field_roi.json
    python define_roi.py --input video.mp4 --frame 90   # use frame 90
"""

import argparse
import json
import sys

import cv2
import numpy as np
VERT_COLOR  = (0, 255, 120)
LINE_COLOR  = (0, 255, 120)
FILL_COLOR  = (0, 255, 120)
FILL_ALPHA  = 0.14
FONT        = cv2.FONT_HERSHEY_DUPLEX
WIN_NAME    = "Define Field ROI  |  Left-click to place points  |  Enter to save"
class ROIEditor:
    def __init__(self, frame: np.ndarray, out_path: str):
        self.base  = frame.copy()
        self.out   = out_path
        self.pts   : list[tuple[int, int]] = []
        self.hover : tuple[int, int]       = (0, 0)
        self.saved = False
    def render(self) -> np.ndarray:
        img = self.base.copy()
        pts = self.pts

        if len(pts) >= 3:
            poly = np.array(pts, np.int32)
            ov   = img.copy()
            cv2.fillPoly(ov, [poly], FILL_COLOR)
            cv2.addWeighted(ov, FILL_ALPHA, img, 1 - FILL_ALPHA, 0, img)

        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], LINE_COLOR, 2, cv2.LINE_AA)

        if pts:
            cv2.line(img, pts[-1], self.hover, (160, 255, 160), 1, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.line(img, pts[0], self.hover, (160, 255, 160), 1, cv2.LINE_AA)

        for i, p in enumerate(pts):
            cv2.circle(img, p, 7, (0, 0, 0), -1)
            cv2.circle(img, p, 5, VERT_COLOR, -1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 6),
                        FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        self._hud(img)
        if self.saved:
            self._saved_banner(img)
        return img

    def _hud(self, img):
        lines = [
            f"Vertices: {len(self.pts)}",
            "Left-click : add point",
            "Right-click: undo",
            "Enter/Space: save",
            "R: reset  |  Q/Esc: quit",
        ]
        bh = len(lines) * 22 + 12
        cv2.rectangle(img, (8, 8), (252, bh), (0, 0, 0), -1)
        cv2.rectangle(img, (8, 8), (252, bh), VERT_COLOR, 1)
        for i, t in enumerate(lines):
            c = (0, 255, 120) if i == 0 else (210, 210, 210)
            cv2.putText(img, t, (14, 28 + i * 22), FONT, 0.43, c, 1, cv2.LINE_AA)

    def _saved_banner(self, img):
        H, W = img.shape[:2]
        msg = "POLYGON SAVED!"
        (tw, th), _ = cv2.getTextSize(msg, FONT, 1.1, 2)
        cx, cy = W // 2, H // 2
        cv2.rectangle(img, (cx - tw // 2 - 18, cy - th - 18),
                      (cx + tw // 2 + 18, cy + 18), (0, 0, 0), -1)
        cv2.putText(img, msg, (cx - tw // 2, cy), FONT, 1.1, (0, 255, 120), 2, cv2.LINE_AA)
    def mouse(self, event, x, y, *_):
        self.hover = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()
    def save(self) -> bool:
        if len(self.pts) < 3:
            print("[roi] Need ≥ 3 points. Keep clicking.")
            return False
        data = {
            "polygon"   : self.pts,
            "n_vertices": len(self.pts),
            "frame_size": [self.base.shape[1], self.base.shape[0]],
        }
        with open(self.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[roi] Saved {len(self.pts)}-vertex polygon → {self.out}")
        self.saved = True
        return True
def grab_frame(path: str, idx: int) -> np.ndarray:
    cap   = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {idx}")
    print(f"[roi] Calibration frame: {idx}/{total}  ({frame.shape[1]}×{frame.shape[0]})")
    return frame


def run(video: str, out: str, frame_idx: int) -> bool:
    frame  = grab_frame(video, frame_idx)
    editor = ROIEditor(frame, out)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(WIN_NAME, editor.mouse)

    while True:
        cv2.imshow(WIN_NAME, editor.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):        # Enter / Space
            if editor.save():
                cv2.imshow(WIN_NAME, editor.render())
                cv2.waitKey(800)
                break
        elif key == ord("r"):
            editor.pts.clear()
        elif key in (ord("q"), 27):
            print("[roi] Aborted.")
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default="field_roi.json")
    ap.add_argument("--frame",  type=int, default=30,
                    help="Frame index to use as calibration image")
    args = ap.parse_args()
    sys.exit(0 if run(args.input, args.output, args.frame) else 1)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("pip install ultralytics")

try:
    import supervision as sv
except ImportError:
    raise ImportError("pip install supervision")

from field_roi import FieldROI
DEFAULT_CONFIG: dict = {
    "model_weights"     : "yolov8l.pt",
    "conf_threshold"    : 0.25,
    "iou_threshold"     : 0.45,
    "imgsz"             : 1280,
    "track_thresh"      : 0.25,   
    "track_buffer"      : 60,
    "match_thresh"      : 0.80,
    "frame_skip"        : 1,
    "output_fps_factor" : 1.0,
    "trail_length"      : 50,
    "draw_trail"        : True,
    "draw_heatmap"      : True,
    "heatmap_alpha"     : 0.40,
    "show_roi_overlay"  : True,
    "show_rejected"     : False,
    "roi_path"          : None,
    "roi_mode"          : "optical_flow",
    "roi_expand_px"     : 15,
    "roi_foot_point"    : True,
    "seg_refit_every"   : 60,
    "person_class_id"   : 0,
    "ball_class_id"     : 32,
}
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0 : (0,   210, 255),   # cyan / team A
    1 : (255,  70,  30),   # red  / team B
    -1: (190, 190, 190),   # grey / referee
}

BALL_COLOR = (255, 255, 255)
def assign_team(cx: float, frame_w: float) -> int:
    """Heuristic: left-of-centre = team 0, right = team 1."""
    return 0 if cx < frame_w / 2 else 1


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=5):
    for p1, p2 in [((x1+r,y1),(x2-r,y1)),((x1+r,y2),(x2-r,y2)),
                   ((x1,y1+r),(x1,y2-r)),((x2,y1+r),(x2,y2-r))]:
        cv2.line(img, p1, p2, color, thickness)
    for centre, start, end in [
        ((x1+r,y1+r),180, 90),((x2-r,y1+r),270,90),
        ((x1+r,y2-r), 90, 90),((x2-r,y2-r),  0,90)]:
        cv2.ellipse(img, centre, (r,r), start, 0, end, color, thickness)


def draw_label(img, text, x, y, color, font_scale=0.52, thickness=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(img,(x-pad,y-th-pad),(x+tw+pad,y+bl), color,-1)
    luma = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    tc   = (15,15,15) if luma > 128 else (240,240,240)
    cv2.putText(img, text,(x,y), font, font_scale, tc, thickness, cv2.LINE_AA)


def draw_trail(img, trail: deque, color):
    pts = list(trail)
    n   = len(pts)
    for i in range(1, n):
        alpha   = i / n
        t_color = tuple(int(c * alpha) for c in color)
        thick   = max(1, int(alpha * 3))
        cv2.line(img, pts[i-1], pts[i], t_color, thick, cv2.LINE_AA)
class HeatmapAccumulator:
    def __init__(self, h: int, w: int, scale: float = 0.25):
        self.s  = scale
        self.sh = int(h * scale)
        self.sw = int(w * scale)
        self.acc = np.zeros((self.sh, self.sw), dtype=np.float32)

    def update(self, cx: int, cy: int):
        sx = int(np.clip(cx * self.s, 2, self.sw - 3))
        sy = int(np.clip(cy * self.s, 2, self.sh - 3))
        self.acc[sy-2:sy+3, sx-2:sx+3] += 1.0

    def render(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if self.acc.max() == 0:
            return frame
        norm  = cv2.normalize(self.acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_up = cv2.resize(norm,(frame.shape[1],frame.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        hm    = cv2.applyColorMap(norm_up, cv2.COLORMAP_JET)
        mask  = norm_up > 10
        out   = frame.copy()
        out[mask] = cv2.addWeighted(frame, 1-alpha, hm, alpha, 0)[mask]
        return out
class StatsCollector:
    def __init__(self):
        self.per_frame    : list  = []
        self.first_seen   : dict  = {}
        self.last_seen    : dict  = {}
        self.rejected_total: int  = 0

    def update(self, frame_idx, tracked, ball_detected, n_rejected):
        n = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        self.per_frame.append((frame_idx, n, int(ball_detected), n_rejected))
        self.rejected_total += n_rejected
        if tracked.tracker_id is None:
            return
        for tid in tracked.tracker_id:
            if tid not in self.first_seen:
                self.first_seen[tid] = frame_idx
            self.last_seen[tid] = frame_idx

    def save(self, path: str):
        data = {
            "total_unique_ids"   : len(self.first_seen),
            "total_roi_rejected" : self.rejected_total,
            "per_frame"          : self.per_frame,
            "track_lifetimes"    : {
                str(tid): {
                    "first": self.first_seen[tid],
                    "last" : self.last_seen[tid],
                    "dur"  : self.last_seen[tid] - self.first_seen[tid],
                }
                for tid in self.first_seen
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[stats] → {path}")
class TrailManager:
    """
    Separates trail update from the tracked detections so that frames where
    a track was not detected (Kalman extrapolation only) do NOT extend the
    trail.  This eliminates the jitter/ghost-streak problem.
    """
    def __init__(self, maxlen: int = 50):
        self.trails     : dict[int, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.detected_ids: set[int]        = set()   # IDs with fresh detections this frame

    def set_detected(self, detected_ids: set[int]):
        """Call at the start of each frame with the set of IDs that were DETECTED."""
        self.detected_ids = detected_ids

    def update(self, tid: int, cx: int, cy: int):
        """Only updates trail if this ID had a real detection this frame."""
        if tid in self.detected_ids:
            self.trails[tid].append((cx, cy))

    def draw(self, img: np.ndarray, tid: int, color):
        t = self.trails[tid]
        if len(t) > 1:
            draw_trail(img, t, color)
class TiledDetector:
    """
    Optionally splits the frame into overlapping tiles and merges detections.
    Effective for players that subtend < 12 px at the original resolution.

    Set cfg['tiled'] = True to enable.  Adds ~3× inference time but
    substantially improves recall of small/distant players.
    """

    OVERLAP = 0.20  # 20% tile overlap to avoid edge misses

    def __init__(self, model, cfg: dict):
        self.model   = model
        self.cfg     = cfg
        self.enabled = cfg.get("tiled", False)
        self.tile_rows = cfg.get("tile_rows", 2)
        self.tile_cols = cfg.get("tile_cols", 3)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (xyxy, confs, class_ids) merged across all tiles.
        """
        if not self.enabled:
            return self._detect_full(frame)

        H, W   = frame.shape[:2]
        th     = int(H / (self.tile_rows * (1 - self.OVERLAP) + self.OVERLAP))
        tw     = int(W / (self.tile_cols * (1 - self.OVERLAP) + self.OVERLAP))
        stride_h = int(th * (1 - self.OVERLAP))
        stride_w = int(tw * (1 - self.OVERLAP))

        all_xyxy, all_conf, all_cls = [], [], []

        for row in range(self.tile_rows):
            for col in range(self.tile_cols):
                y0 = row * stride_h
                x0 = col * stride_w
                y1 = min(y0 + th, H)
                x1 = min(x0 + tw, W)
                tile = frame[y0:y1, x0:x1]

                bx, cf, cl = self._detect_full(tile)
                if len(bx):
                    bx[:, [0, 2]] += x0
                    bx[:, [1, 3]] += y0
                    all_xyxy.append(bx)
                    all_conf.append(cf)
                    all_cls.append(cl)

        if not all_xyxy:
            return np.empty((0,4)), np.empty((0,)), np.empty((0,))

        xyxy = np.concatenate(all_xyxy)
        conf = np.concatenate(all_conf)
        cls  = np.concatenate(all_cls)
        keep = cv2.dnn.NMSBoxes(
            xyxy[:, :4].tolist(),
            conf.tolist(),
            self.cfg["conf_threshold"],
            self.cfg["iou_threshold"],
        )
        if isinstance(keep, np.ndarray) and len(keep):
            keep = keep.flatten()
            return xyxy[keep], conf[keep], cls[keep]
        return np.empty((0,4)), np.empty((0,)), np.empty((0,))

    def _detect_full(self, frame):
        res   = self.model.predict(
            frame,
            conf    = self.cfg["conf_threshold"],
            iou     = self.cfg["iou_threshold"],
            imgsz   = self.cfg["imgsz"],
            classes = [self.cfg["person_class_id"], self.cfg["ball_class_id"]],
            verbose = False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0,4)), np.empty((0,)), np.empty((0,))
        return (
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        )
class FootballTracker:
    """
    Production-grade pipeline:
      YOLOv8l (imgsz=1280) → ROI pre-filter → ByteTrack → annotation

    Problem → solution mapping:
      [1] ID instability   → ByteTrack + track_buffer=60 + DetectionID guard
      [2] Missing players  → yolov8l + imgsz=1280 + conf=0.25 + optional tiles
      [3] Overlap tracking → ByteTrack two-stage association
      [4] ROI camera drift → optical-flow homography warp each frame
      [5] Noisy trails     → TrailManager only appends on detected (not predicted) frames
      [6] FPS tradeoff     → frame_skip config, imgsz config
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        print(f"[init] Model       : {cfg['model_weights']}")
        print(f"[init] imgsz       : {cfg['imgsz']}")
        print(f"[init] conf        : {cfg['conf_threshold']}")
        print(f"[init] track_buffer: {cfg['track_buffer']}")
        print(f"[init] ROI mode    : {cfg['roi_mode']}")

        self.model    = YOLO(cfg["model_weights"])
        self.detector = TiledDetector(self.model, cfg)

        self.tracker = sv.ByteTrack(
            track_activation_threshold = cfg["track_thresh"],
            lost_track_buffer          = cfg["track_buffer"],
            minimum_matching_threshold = cfg["match_thresh"],
            frame_rate                 = 30,
        )

        self.roi = FieldROI(
            roi_source         = cfg.get("roi_path"),
            mode               = cfg.get("roi_mode", "optical_flow"),
            foot_point_mode    = cfg.get("roi_foot_point", True),
            expand_px          = cfg.get("roi_expand_px", 15),
            segmentation_refit = cfg.get("seg_refit_every", 60),
        )

        self.trails   = TrailManager(cfg["trail_length"])
        self.team_map : dict[int, int] = {}
        self.stats    = StatsCollector()
    def _rescale_roi(self, roi_path, W, H):
        if not roi_path or not Path(roi_path).exists():
            return
        with open(roi_path) as f:
            data = json.load(f)
        stored = data.get("frame_size")
        if stored and (stored[0] != W or stored[1] != H):
            print(f"[roi] Rescaling {stored[0]}×{stored[1]} → {W}×{H}")
            self.roi = FieldROI(
                roi_source         = roi_path,
                mode               = self.cfg.get("roi_mode", "optical_flow"),
                scale_from         = tuple(stored),
                target_size        = (W, H),
                foot_point_mode    = self.cfg.get("roi_foot_point", True),
                expand_px          = self.cfg.get("roi_expand_px", 15),
                segmentation_refit = self.cfg.get("seg_refit_every", 60),
            )
    def process_frame(
        self,
        frame    : np.ndarray,
        frame_idx: int,
        heatmap  : HeatmapAccumulator,
    ) -> np.ndarray:
        H, W = frame.shape[:2]
        self.roi.update_camera(frame)
        all_xyxy, all_conf, all_cls = self.detector.detect(frame)

        pid = self.cfg["person_class_id"]
        bid = self.cfg["ball_class_id"]

        person_mask = (all_cls == pid)
        ball_mask   = (all_cls == bid)
        n_rejected   = 0
        outside_xyxy = np.empty((0, 4))

        if person_mask.any():
            p_xyxy  = all_xyxy[person_mask]
            p_conf  = all_conf[person_mask]

            inside = np.array([
                self.roi.is_inside(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                for r in p_xyxy
            ], dtype=bool)

            outside_xyxy = p_xyxy[~inside]
            n_rejected   = int((~inside).sum())
            field_xyxy   = p_xyxy[inside]
            field_conf   = p_conf[inside]
        else:
            field_xyxy = np.empty((0, 4))
            field_conf = np.empty((0,))
        if len(field_xyxy) > 0:
            det_sv = sv.Detections(
                xyxy      = field_xyxy,
                confidence= field_conf,
                class_id  = np.zeros(len(field_xyxy), dtype=int),
            )
            tracked = self.tracker.update_with_detections(det_sv)
        else:
            tracked = self.tracker.update_with_detections(sv.Detections.empty())
        fresh_ids: set[int] = set()
        if tracked.tracker_id is not None and len(field_xyxy) > 0:
            for i, tid in enumerate(tracked.tracker_id):
                tx1, ty1, tx2, ty2 = tracked.xyxy[i]
                ious = _batch_iou(
                    np.array([[tx1, ty1, tx2, ty2]]),
                    field_xyxy
                )
                if ious.max() > 0.3:
                    fresh_ids.add(int(tid))
        self.trails.set_detected(fresh_ids)

        self.stats.update(frame_idx, tracked, ball_mask.any(), n_rejected)
        annotated = (
            self.roi.draw_overlay(frame)
            if self.cfg.get("show_roi_overlay", True)
            else frame.copy()
        )
        if self.cfg.get("show_rejected", False):
            for bx in outside_xyxy:
                self.roi.draw_rejected_box(annotated,
                    int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tid    = int(tid)
                if tid not in self.team_map:
                    self.team_map[tid] = assign_team(cx, W)
                color = TEAM_COLORS[self.team_map[tid]]
                self.trails.update(tid, cx, cy)
                if self.cfg["draw_trail"]:
                    self.trails.draw(annotated, tid, color)

                heatmap.update(cx, cy)
                draw_rounded_rect(annotated, x1, y1, x2, y2, color)
                tag = f"P#{tid}" if tid in fresh_ids else f"P#{tid}~"
                draw_label(annotated, tag, x1, y1 - 2, color)
        if ball_mask.any():
            for bx in all_xyxy[ball_mask]:
                bx1, by1, bx2, by2 = bx.astype(int)
                bcx, bcy = (bx1+bx2)//2, (by1+by2)//2
                cv2.circle(annotated, (bcx, bcy), 11, (0,0,0),     2)
                cv2.circle(annotated, (bcx, bcy),  9, BALL_COLOR,   2)
                cv2.circle(annotated, (bcx, bcy),  4, (255,220,0), -1)
                draw_label(annotated, "Ball", bx1, by1-2, (230,230,0))
        n_tracked  = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        fresh_ct   = len(fresh_ids)
        roi_status = {"optical_flow":"OF","segmentation":"SEG","static":"FIX"}.get(
            self.cfg.get("roi_mode","static"), "?")
        hud_lines = [
            f"Frame   : {frame_idx:>5}",
            f"Tracked : {n_tracked:>5}  (det:{fresh_ct})",
            f"Filtered: {n_rejected:>5}",
            f"IDs seen: {len(self.stats.first_seen):>4}",
            f"ROI     : {roi_status}",
        ]
        for li, txt in enumerate(hud_lines):
            draw_label(annotated, txt, 12, 28 + li*26, (25,25,25), font_scale=0.46)

        return annotated
    def run(self, inp: str, out_vid: str, out_hm: str, out_stats: str):
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {inp}")

        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[video] {W}×{H} @ {fps:.1f}fps | {total} frames")

        self._rescale_roi(self.cfg.get("roi_path"), W, H)

        writer = cv2.VideoWriter(
            out_vid, cv2.VideoWriter_fourcc(*"mp4v"),
            fps * self.cfg["output_fps_factor"], (W, H))

        heatmap   = HeatmapAccumulator(H, W)
        frame_idx = 0
        t0        = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.cfg["frame_skip"] == 0:
                annotated = self.process_frame(frame, frame_idx, heatmap)
            else:
                annotated = frame

            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 60 == 0:
                el       = time.time() - t0
                fps_proc = frame_idx / el
                eta      = (total - frame_idx) / max(fps_proc, 1)
                print(f"  [{frame_idx/max(total,1)*100:5.1f}%] "
                      f"frame {frame_idx}/{total} | "
                      f"{fps_proc:.1f}fps | ETA {eta:.0f}s | "
                      f"IDs:{len(self.stats.first_seen)}")

        cap.release()
        writer.release()
        print(f"\n[done] Video  → {out_vid}")

        if self.cfg["draw_heatmap"]:
            hm = heatmap.render(np.zeros((H, W, 3), np.uint8), alpha=1.0)
            cv2.imwrite(out_hm, hm)
            print(f"[done] Heatmap→ {out_hm}")

        self.stats.save(out_stats)

        el = time.time() - t0
        print(f"\n{'═'*55}")
        print(f"  Frames processed : {frame_idx} in {el:.1f}s  ({frame_idx/el:.1f}fps)")
        print(f"  Unique player IDs: {len(self.stats.first_seen)}")
        print(f"  ROI rejected     : {self.stats.rejected_total} detections")
        print(f"{'═'*55}")
def _batch_iou(boxA: np.ndarray, boxB: np.ndarray) -> np.ndarray:
    """
    Compute IoU between each row of boxA and each row of boxB.
    Returns (len(boxA), len(boxB)) matrix.
    """
    if len(boxB) == 0:
        return np.zeros((len(boxA), 0))
    ax1,ay1,ax2,ay2 = boxA[:,0],boxA[:,1],boxA[:,2],boxA[:,3]
    bx1,by1,bx2,by2 = boxB[:,0],boxB[:,1],boxB[:,2],boxB[:,3]
    inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
    inter_y1 = np.maximum(ay1[:,None], by1[None,:])
    inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
    inter_y2 = np.minimum(ay2[:,None], by2[None,:])
    inter_w  = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h  = np.clip(inter_y2 - inter_y1, 0, None)
    inter    = inter_w * inter_h
    areaA    = (ax2 - ax1) * (ay2 - ay1)
    areaB    = (bx2 - bx1) * (by2 - by1)
    union    = areaA[:,None] + areaB[None,:] - inter
    return inter / np.where(union == 0, 1, union)
def parse_args():
    p = argparse.ArgumentParser(
        description="Production Football Tracker — YOLOv8l + ByteTrack + Dynamic ROI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick-start:
  python define_roi.py   --input video.mp4 --output field_roi.json
  python track_football.py --input video.mp4 --roi field_roi.json
  python analytics.py    --stats output_stats.json

Speed vs accuracy presets:
  Fast (CPU):    --model yolov8n.pt --imgsz 640  --skip 2
  Balanced:      --model yolov8m.pt --imgsz 960  --skip 1   [default-ish]
  Best quality:  --model yolov8l.pt --imgsz 1280 --skip 1   [default]
  Max (GPU):     --model yolov8x.pt --imgsz 1280 --tiled

ROI modes:
  optical_flow  — polygon warps with camera using LK optical flow  [default]
  segmentation  — re-estimates boundary from green grass every 60 frames
  static        — fixed polygon (use only for tripod/fixed cameras)
""")
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   default="output_tracked.mp4")
    p.add_argument("--heatmap",  default="output_heatmap.png")
    p.add_argument("--stats",    default="output_stats.json")
    p.add_argument("--model",    default=DEFAULT_CONFIG["model_weights"],
                   help="YOLOv8 weights: yolov8n/s/m/l/x.pt")
    p.add_argument("--conf",     type=float, default=DEFAULT_CONFIG["conf_threshold"])
    p.add_argument("--imgsz",    type=int,   default=DEFAULT_CONFIG["imgsz"],
                   help="Inference resolution (640/960/1280). Higher = better recall.")
    p.add_argument("--tiled",    action="store_true",
                   help="Enable tiled inference for very small distant players")
    p.add_argument("--track-buffer", type=int, default=DEFAULT_CONFIG["track_buffer"],
                   help="Frames to keep a lost track alive (default: 60 = 2s at 30fps)")
    p.add_argument("--skip",     type=int, default=DEFAULT_CONFIG["frame_skip"])
    p.add_argument("--roi",      default=None,
                   help="Path to field_roi.json from define_roi.py")
    p.add_argument("--roi-mode", default=DEFAULT_CONFIG["roi_mode"],
                   choices=["optical_flow","segmentation","static"],
                   help="Camera compensation mode for the ROI polygon")
    p.add_argument("--roi-expand",   type=int, default=DEFAULT_CONFIG["roi_expand_px"])
    p.add_argument("--show-rejected",action="store_true")
    p.add_argument("--hide-roi",     action="store_true")
    p.add_argument("--no-trail",     action="store_true")
    p.add_argument("--no-heatmap",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()
    cfg.update({
        "model_weights"   : args.model,
        "conf_threshold"  : args.conf,
        "track_thresh"    : args.conf,   # keep in sync
        "imgsz"           : args.imgsz,
        "track_buffer"    : args.track_buffer,
        "frame_skip"      : args.skip,
        "draw_trail"      : not args.no_trail,
        "draw_heatmap"    : not args.no_heatmap,
        "roi_path"        : args.roi,
        "roi_mode"        : args.roi_mode,
        "roi_expand_px"   : args.roi_expand,
        "show_rejected"   : args.show_rejected,
        "show_roi_overlay": not args.hide_roi,
        "tiled"           : args.tiled,
    })
    FootballTracker(cfg).run(args.input, args.output, args.heatmap, args.stats)


if __name__ == "__main__":
    main()
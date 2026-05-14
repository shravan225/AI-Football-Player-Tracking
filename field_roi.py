"""
field_roi.py  —  Dynamic Field ROI with Homography-based Camera Adaptation
===========================================================================
Solves the "ROI drifts when camera pans/zooms" problem by tracking the
field polygon across frames using:

  Option A — Optical-flow warp (default, zero dependencies beyond OpenCV)
    Lucas-Kanade sparse optical flow tracks ~100 corner points frame-to-frame.
    The resulting per-frame homography is used to warp the polygon vertices,
    keeping the green boundary glued to the pitch even during camera motion.

  Option B — Pitch-line homography (advanced, more accurate)
    Detects green-field HSV mask, finds major line segments via HoughLinesP,
    and re-estimates the field boundary every N frames.

  Option C — Static polygon (original behaviour, no camera compensation)
    Used when camera is fixed or when optical flow is disabled via flag.

Foot-point convention
---------------------
We test (bottom-centre) of each bbox against the polygon — where feet touch
the ground — rather than the box centre, which can be above the touchline
for players standing at the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
class _OpticalFlowWarper:
    """
    Tracks sparse feature points between consecutive frames.
    Returns a 3×3 homography that describes the camera motion,
    used to warp the field polygon to stay aligned with the pitch.
    """

    FEATURE_PARAMS = dict(
        maxCorners   = 150,
        qualityLevel = 0.01,
        minDistance  = 10,
        blockSize    = 7,
    )
    LK_PARAMS = dict(
        winSize  = (21, 21),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    MIN_INLIERS   = 15
    REFRESH_EVERY = 20

    def __init__(self):
        self._prev_gray : np.ndarray | None = None
        self._prev_pts  : np.ndarray | None = None  # shape (N,1,2) float32
        self._frame_cnt = 0

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Call once per frame. Returns a 3×3 homography (prev→curr) or None
        if motion cannot be estimated (first frame, or too few inliers).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_out = None

        if self._prev_gray is not None and self._prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **self.LK_PARAMS)

            if curr_pts is not None and status is not None:
                good_prev = self._prev_pts[status.ravel() == 1]
                good_curr = curr_pts[status.ravel() == 1]

                if len(good_prev) >= self.MIN_INLIERS:
                    H_out, inlier_mask = cv2.findHomography(
                        good_prev, good_curr, cv2.RANSAC, 3.0)
                    if H_out is None or (inlier_mask is not None and
                                         inlier_mask.sum() < self.MIN_INLIERS):
                        H_out = None
                    else:
                        self._prev_pts = good_curr[inlier_mask.ravel() == 1].reshape(-1, 1, 2)
        needs_refresh = (
            self._prev_gray is None
            or self._frame_cnt % self.REFRESH_EVERY == 0
            or (self._prev_pts is not None and len(self._prev_pts) < 30)
        )
        if needs_refresh:
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEATURE_PARAMS)
            self._prev_pts = pts  # may be None if no features found

        self._prev_gray = gray
        self._frame_cnt += 1
        return H_out
class _FieldSegmenter:

    GRASS_LOWER = np.array([30,  40,  40], dtype=np.uint8)
    GRASS_UPPER = np.array([85, 255, 255], dtype=np.uint8)

    def __init__(self, refit_every: int = 60):
        self.refit_every = refit_every
        self._last_poly: np.ndarray | None = None
        self._frame_cnt = 0

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        
        self._frame_cnt += 1
        if self._frame_cnt % self.refit_every != 1:
            return self._last_poly   # reuse cached result

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.GRASS_LOWER, self.GRASS_UPPER)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._last_poly
        largest = max(contours, key=cv2.contourArea)
        H, W    = frame.shape[:2]
        if cv2.contourArea(largest) < 0.05 * H * W:
            return self._last_poly  # too small — unreliable

        hull = cv2.convexHull(largest)
        eps  = 0.02 * cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2)
        self._last_poly = poly
        return poly
class FieldROI:
    ROI_LINE_COLOR   = (0, 220, 110)
    ROI_FILL_COLOR   = (0, 220, 110)
    ROI_FILL_ALPHA   = 0.08
    ROI_VERTEX_COLOR = (255, 255, 0)
    REJECT_COLOR     = (60,  60, 200)

    def __init__(
        self,
        roi_source=None,
        mode: str = "optical_flow",
        scale_from: tuple | None = None,
        target_size: tuple | None = None,
        foot_point_mode: bool = True,
        expand_px: int = 12,
        segmentation_refit: int = 60,
    ):
        self.foot_point_mode = foot_point_mode
        self.mode            = mode
        self._enabled        = False
        self._base_polygon   : np.ndarray | None = None  # shape (N,2) float32 — original
        self._cur_polygon    : np.ndarray | None = None  # shape (N,2) int32   — current frame
        self._of_warper   = _OpticalFlowWarper() if mode == "optical_flow"   else None
        self._seg_helper  = _FieldSegmenter(segmentation_refit) if mode == "segmentation" else None

        if roi_source is None and mode == "segmentation":
            self._enabled = True
            print("[roi] Segmentation mode — field boundary estimated from green mask each frame")
            return

        if roi_source is None:
            print("[roi] No ROI source — pass-through mode (all detections kept)")
            return
        if isinstance(roi_source, (str, Path)):
            pts = self._load_json(str(roi_source))
        elif isinstance(roi_source, (list, np.ndarray)):
            pts = [tuple(p) for p in roi_source]
        else:
            raise TypeError(f"roi_source must be path, list, or None. Got {type(roi_source)}")

        if len(pts) < 3:
            raise ValueError("ROI polygon must have ≥ 3 vertices.")

        poly = np.array(pts, dtype=np.float32)
        if scale_from and target_size:
            sw = target_size[0] / scale_from[0]
            sh = target_size[1] / scale_from[1]
            poly[:, 0] *= sw
            poly[:, 1] *= sh
            print(f"[roi] Rescaled polygon {scale_from} → {target_size}  "
                  f"(sx={sw:.3f}, sy={sh:.3f})")

        if expand_px:
            poly = self._expand_polygon(poly, expand_px)

        self._base_polygon = poly
        self._cur_polygon  = poly.astype(np.int32)
        self._enabled      = True
        print(f"[roi] Loaded — {len(poly)} vertices | mode={mode} | "
              f"foot_point={foot_point_mode} | expand={expand_px}px")

    def update_camera(self, frame: np.ndarray) -> None:
        """
        Update the internal polygon estimate to compensate for camera motion.
        Must be called exactly once per frame BEFORE any is_inside() calls.
        """
        if not self._enabled:
            return

        if self.mode == "optical_flow" and self._of_warper is not None:
            H = self._of_warper.update(frame)
            if H is not None and self._cur_polygon is not None:
                self._cur_polygon = self._warp_polygon(self._cur_polygon.astype(np.float32), H)

        elif self.mode == "segmentation" and self._seg_helper is not None:
            poly = self._seg_helper.update(frame)
            if poly is not None:
                self._cur_polygon = poly.astype(np.int32)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def is_inside(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        
        if not self._enabled or self._cur_polygon is None:
            return True
        px, py = self._test_point(x1, y1, x2, y2)
        result = cv2.pointPolygonTest(
            self._cur_polygon, (float(px), float(py)), measureDist=False)
        return result >= 0

    def filter_detections(
        self,
        xyxy: np.ndarray,
        *extra_arrays: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """
        Filter an (N,4) xyxy array in one vectorised call.

        Returns (filtered_xyxy, *filtered_extras)
        All extra arrays are filtered with the same boolean mask.
        """
        if not self._enabled or self._cur_polygon is None or len(xyxy) == 0:
            return (xyxy, *extra_arrays)

        keep = np.array([
            self.is_inside(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            for r in xyxy
        ], dtype=bool)

        return (xyxy[keep], *(arr[keep] for arr in extra_arrays))

    def draw_overlay(
        self,
        frame: np.ndarray,
        show_fill    : bool = True,
        show_vertices: bool = True,
        show_label   : bool = True,
    ) -> np.ndarray:
        
        if not self._enabled or self._cur_polygon is None:
            return frame

        img  = frame.copy()
        poly = self._cur_polygon.reshape((-1, 1, 2))

        if show_fill:
            overlay = img.copy()
            cv2.fillPoly(overlay, [self._cur_polygon], self.ROI_FILL_COLOR)
            cv2.addWeighted(overlay, self.ROI_FILL_ALPHA, img, 1 - self.ROI_FILL_ALPHA, 0, img)

        cv2.polylines(img, [poly], isClosed=True,
                      color=self.ROI_LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)

        if show_vertices:
            for pt in self._cur_polygon:
                cv2.circle(img, tuple(pt.tolist()), 5, self.ROI_VERTEX_COLOR, -1)

        if show_label:
            mode_tag = {"optical_flow": "OF", "segmentation": "SEG", "static": "STATIC"}.get(
                self.mode, "?")
            M = self._cur_polygon.mean(axis=0).astype(int)
            cv2.putText(img, f"Field ROI [{mode_tag}]", tuple(M.tolist()),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, self.ROI_LINE_COLOR, 1, cv2.LINE_AA)
        return img

    def draw_rejected_box(self, frame: np.ndarray, x1, y1, x2, y2):
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.REJECT_COLOR, 1)
        cv2.putText(frame, "out", (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, self.REJECT_COLOR, 1)

    def get_polygon_points(self) -> list[tuple[int, int]]:
        if self._cur_polygon is None:
            return []
        return [tuple(p.tolist()) for p in self._cur_polygon]

    def _test_point(self, x1, y1, x2, y2) -> tuple[int, int]:
        if self.foot_point_mode:
            return (x1 + x2) // 2, y2
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def _load_json(path: str) -> list[tuple[int, int]]:
        if not Path(path).exists():
            raise FileNotFoundError(f"ROI file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        pts = data.get("polygon") or data.get("points") or data
        return [tuple(p) for p in pts]

    @staticmethod
    def _expand_polygon(poly: np.ndarray, px: int) -> np.ndarray:
        centroid   = poly.mean(axis=0)
        directions = poly - centroid
        norms      = np.linalg.norm(directions, axis=1, keepdims=True)
        norms      = np.where(norms == 0, 1, norms)
        return poly + (directions / norms) * px

    @staticmethod
    def _warp_polygon(poly: np.ndarray, H: np.ndarray) -> np.ndarray:
        
        pts = poly.reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2).astype(np.int32)
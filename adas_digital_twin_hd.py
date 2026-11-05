#!/usr/bin/env python3
"""
ADAS DIGITAL TWIN - HIGH-GRAPHICS VIDEO REGENERATION  (v3)
==========================================================
Run in PyCharm -> popup 1: select road video -> popup 2: select output folder
-> popup 3: lane calibration (YES = automatic, NO = manual 4 clicks, CANCEL = defaults).

v3 lane engine
  * Automatic camera calibration: vanishing point (RANSAC on road lines) + scale from lane width.
  * Shape filter on bird's-eye-view (BEV) blobs: only long, thin, longitudinal paint is a lane line.
    Crosswalk stripes, stop lines, arrows and text are rejected (red in the BEV panel).
  * Every lane line is tracked independently with a persistent ID (up to 3 lanes each side):
    lane count, ego lane number, actor lane index from real lines.
  * Lane change = a tracked line crossing the ego position with hysteresis (no width-jump guessing).
  * Junction mode (crosswalk / stop line / markings vanish): lines coasted, no lane-change
    events, crosswalk drawn in the twin, JUNCTION chip.
Other features: 3-D super-sampled twin (light/dark theme), YOLO11 + ByteTrack with class voting,
cut-in / cut-out / TTC / lead vehicle, traffic-light state, ego speed estimate, CSV/JSON exports.

Install (once):
    python -m pip install numpy opencv-python ultralytics lap

Engineering note: monocular video has no true metric scale. Distances depend on the calibration.
Use it as a reconstruction / visualisation tool, not as a safety pass/fail system.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# =====================================================================
# USER SETTINGS
# =====================================================================
MODEL_WEIGHTS = "yolo11s.pt"        # yolo11n/s/m/l.pt or your custom .pt (e.g. tractor, auto-rickshaw)
FALLBACK_WEIGHTS = "yolov8s.pt"     # used if MODEL_WEIGHTS cannot be loaded
DET_CONF, DET_IOU, IMG_SIZE = 0.30, 0.50, 960
TRACKER_CFG = "bytetrack.yaml"
KEEP_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
                "traffic light", "stop sign", "tractor", "auto", "autorickshaw"}  # None = keep all

LANE_WIDTH_M = 3.3                  # nominal lane width (city ~3.0-3.3, highway ~3.5-3.75)
NEAR_M, FAR_M = 6.0, 30.0           # manual calibration: ground distance of bottom / top points
DEFAULT_TRAPEZOID = [(0.20, 0.92), (0.82, 0.92), (0.565, 0.65), (0.445, 0.65)]  # BL, BR, TR, TL (fractions)
ASK_LANE_CALIBRATION = True         # popup: AUTO calibration / manual 4-point click / defaults
AUTO_CALIB_SECONDS = 20             # video span sampled for automatic calibration
CAMERA_HFOV_DEG = 70.0              # horizontal field of view (dashcam ~60-80 deg, after undistortion)
CAMERA_HEIGHT_M = 1.40              # camera height above road (initial; auto-scaled by lane width)
AUTO_SCALE_FROM_LANE_WIDTH = True   # rescale distances so the ego lane width = LANE_WIDTH_M

CAMERA_MODEL = "pinhole"            # "pinhole" | "fisheye"
CAMERA_K = None                     # e.g. [[fx,0,cx],[0,fy,cy],[0,0,1]] -> enables undistortion
CAMERA_D = None                     # pinhole: k1,k2,p1,p2,k3 | fisheye: k1..k4

EGO_SPEED_KPH = None                # constant override; None = estimate from video
THEME = "light"                     # "light" (Tesla-style) | "dark"
SUPERSAMPLE = 2                     # 1 = faster, 2 = smoother graphics
SHOW_LABELS = True
SHOW_PATH_PLANNING = False           # Hide path ribbon, chevrons and target-lane shading
SHOW_PREVIEW = True                 # q = quit, p = pause
MAX_SECONDS = None                  # limit processing length (None = full video)
FRAME_STRIDE = 1
OUT_W, OUT_H = 1920, 1080

LOG = logging.getLogger("adas_twin")
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX
VEHICLE_KINDS = {"car", "truck", "bus", "motorcycle", "bicycle", "tractor", "auto"}


# =====================================================================
# Small helpers
# =====================================================================
def pv(c, d):
    return c[0] * d * d + c[1] * d + c[2]


def pv_ext(c, d, dmax):
    """Quadratic inside fitted range, linear extrapolation beyond dmax."""
    d = np.asarray(d, dtype=np.float64)
    inside = pv(c, np.minimum(d, dmax))
    slope = 2 * c[0] * dmax + c[1]
    return inside + np.where(d > dmax, slope * (d - dmax), 0.0)


def put(img, text, x, y, scale=0.5, color=(235, 235, 235), thick=1, font=FONT):
    cv2.putText(img, str(text), (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def text_w(text, scale, thick=1, font=FONT):
    return cv2.getTextSize(str(text), font, scale, thick)[0][0]


def rrect(img, x, y, w, h, r, color, thick=-1):
    x, y, w, h, r = int(x), int(y), int(w), int(h), int(max(1, min(r, w // 2, h // 2)))
    if thick < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thick, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thick, cv2.LINE_AA)
        for cx, cy, a in ((x + r, y + r, 180), (x + w - r, y + r, 270), (x + r, y + h - r, 90), (x + w - r, y + h - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thick, cv2.LINE_AA)


def blend_rect(img, x, y, w, h, color, alpha):
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def kind_of(name: str) -> str:
    n = name.lower()
    if "traffic light" in n: return "light"
    if "stop sign" in n: return "sign"
    if "truck" in n or "lorry" in n: return "truck"
    if "bus" in n: return "bus"
    if "motor" in n or "scooter" in n: return "motorcycle"
    if "bicycle" in n or "cycle" in n: return "bicycle"
    if "person" in n or "pedestrian" in n or "rider" in n: return "person"
    if "tractor" in n: return "tractor"
    if "auto" in n or "rickshaw" in n: return "auto"
    return "car"


CLASS_COLORS = {"car": (90, 205, 60), "truck": (235, 140, 45), "bus": (40, 165, 255), "motorcycle": (210, 90, 220),
                "bicycle": (220, 200, 60), "person": (70, 70, 240), "light": (0, 215, 255), "sign": (60, 60, 210),
                "tractor": (60, 190, 190), "auto": (0, 200, 255)}


def make_undistorter():
    if CAMERA_K is None or CAMERA_D is None:
        return lambda f: f
    K = np.asarray(CAMERA_K, np.float64)
    D = np.asarray(CAMERA_D, np.float64).reshape(-1, 1)
    cache = {}

    def und(frame):
        h, w = frame.shape[:2]
        if (w, h) not in cache:
            if CAMERA_MODEL == "fisheye":
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            else:
                m1, m2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            cache[(w, h)] = (m1, m2)
        m1, m2 = cache[(w, h)]
        return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
    return und


# =====================================================================
# Ground geometry (image <-> metric ground <-> BEV raster)
# =====================================================================
class GroundGeometry:
    """Ego frame: d = longitudinal (m, forward), lat = lateral (m, left positive)."""

    def __init__(self, fw, fh, trap, near, far, lane_w, H=None):
        self.fw, self.fh, self.near, self.far = fw, fh, near, far
        if H is not None:
            self.H = np.asarray(H, np.float64)
        else:
            src = np.float32([[x * fw, y * fh] for x, y in trap])
            dst = np.float32([[near, lane_w / 2], [near, -lane_w / 2], [far, -lane_w / 2], [far, lane_w / 2]])
            self.H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        self.Hinv = np.linalg.inv(self.H)
        ref = self.H @ np.array([fw / 2, fh - 1.0, 1.0])
        self.wsign = 1.0 if ref[2] > 0 else -1.0
        ref2 = self.Hinv @ np.array([near, 0.0, 1.0])
        self.wsign_inv = 1.0 if ref2[2] > 0 else -1.0
        self.lat_half = 3.3 * lane_w          # wide BEV -> up to 3 lanes each side
        self.d0, self.d1 = max(1.0, near - 3.0), far + 12.0
        self.bw, self.bh = 560, 640
        self.mx = self.bw / (2 * self.lat_half)
        self.my = self.bh / (self.d1 - self.d0)
        A = np.array([[0, -self.mx, self.lat_half * self.mx], [-self.my, 0, self.d1 * self.my], [0, 0, 1.0]])
        self.M = A @ self.H

    def img_to_ground(self, u, v):
        q = self.H @ np.array([u, v, 1.0])
        if q[2] * self.wsign <= 1e-12:
            return None
        d, lat = q[0] / q[2], q[1] / q[2]
        if not (0.3 < d < 250 and abs(lat) < 60):
            return None
        return float(d), float(lat)

    def ground_to_img(self, d, lat):
        d = np.asarray(d, np.float64); lat = np.asarray(lat, np.float64)
        P = self.Hinv @ np.vstack([d, lat, np.ones_like(d)])
        ok = P[2] * self.wsign_inv > 1e-12
        z = np.where(np.abs(P[2]) < 1e-12, 1e-12, P[2])
        return np.stack([P[0] / z, P[1] / z], 1), ok

    def bev_to_metric(self, v, u):
        return self.d1 - v / self.my, self.lat_half - u / self.mx

    def metric_to_bev(self, d, lat):
        return (self.lat_half - np.asarray(lat)) * self.mx, (self.d1 - np.asarray(d)) * self.my


# =====================================================================
# Lane detection v3: multi-line tracking, marking-shape filter, junction handling
# =====================================================================
class LaneDetector:
    """Detects and tracks ALL longitudinal lane lines independently (persistent IDs)."""

    def __init__(self, geo, lane_w: float, fps: float = 30.0):
        self.g, self.w_nom = geo, lane_w
        self.w_est = lane_w
        self.fps = fps
        self.tracks = []
        self.next_id = 1
        self.ego = (None, None)          # (left line id, right line id)
        self.cl = self.cr = None
        self.conf, self.switch, self.crossed_type = 0.0, 0, "dashed"
        self.frame = 0
        self.junction_until = -1
        self.no_line_frames = 0
        self.crosswalk_d = self.stopline_d = None
        self.d_ref = geo.near + 2.0
        k = max(9, int(0.8 * geo.mx)) | 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        self.last_binary = self.last_yellow = self.last_reject = None
        self.visible = None

    # ---------------- marking mask --------------------------------------------------
    def _binary(self, bev, block):
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        Hh, L, S = cv2.split(hls)
        valid = (cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) > 6).astype(np.uint8)
        valid = (cv2.erode(valid, np.ones((7, 7), np.uint8)) > 0) & (block == 0)
        Lb = cv2.GaussianBlur(L, (3, 3), 0)
        th = cv2.morphologyEx(Lb, cv2.MORPH_TOPHAT, self.kernel)
        if valid.sum() > 500:
            t1 = max(14.0, float(np.percentile(th[valid], 94)))
            lref = float(np.percentile(Lb[valid], 55))
        else:
            t1, lref = 25.0, 100.0
        bright = (th > t1) & (Lb > lref + 10)
        yellow = (Hh >= 12) & (Hh <= 38) & (S >= 60) & (L >= 70) & valid
        binary = ((bright | (yellow & (th > 0.5 * t1))) & valid).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        self.visible = valid
        return binary, yellow.astype(np.uint8)

    def _shape_filter(self, binary):
        """Keep long thin longitudinal blobs; detect crosswalks and stop lines."""
        g = self.g
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros(n, np.uint8)
        stripes, transverse = [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 8:
                continue
            sub = lab[y:y + h, x:x + w] == i
            vs, us = np.nonzero(sub)
            if len(vs) < 5:
                continue
            lat = (us + x) / g.mx
            dd = (vs + y) / g.my
            cov = np.cov(np.vstack([dd, lat]))
            ev, evec = np.linalg.eigh(cov)
            length = math.sqrt(12 * max(ev[1], 1e-9))
            width = math.sqrt(12 * max(ev[0], 1e-9))
            major = evec[:, 1]
            ang = math.degrees(math.atan2(abs(major[1]), abs(major[0])))   # 0 = along driving direction
            dc = g.d1 - (y + h / 2) / g.my
            if ang < 22 and width < 0.42 and length > 0.9:
                keep[i] = 1                                                  # lane paint
            elif ang < 30 and 0.38 <= width <= 1.3 and length > 1.0:
                stripes.append((dc, g.lat_half - (x + w / 2) / g.mx))       # crosswalk stripe
            elif ang > 55 and length > 1.6:
                transverse.append(dc)                                        # stop line / transverse
        filt = (keep[lab] > 0).astype(np.uint8) * 255
        reject = ((lab > 0) & (keep[lab] == 0)).astype(np.uint8) * 255
        cross_d = None
        if len(stripes) >= 4:
            ds = np.array([s[0] for s in stripes])
            for d0 in ds:
                grp = np.abs(ds - d0) < 2.5
                if grp.sum() >= 4:
                    lats = np.sort([s[1] for s, k in zip(stripes, grp) if k])
                    gaps = np.diff(lats)
                    # zebra = >=4 stripes with ~1 m pitch (arrows are one per lane, ~3 m apart)
                    if lats[-1] - lats[0] > 2.5 and np.median(gaps) < 1.7:
                        cross_d = float(np.median(ds[grp]))
                        break
        stop_d = float(np.median(transverse)) if len(transverse) >= 1 else None
        return filt, reject, cross_d, stop_d

    # ---------------- fitting helpers ----------------------------------------------
    def _coverage(self, d, line_fn=None):
        """Fraction of VISIBLE bins with paint (occlusion does not turn solid into dashed)."""
        if len(d) == 0:
            return 0.0
        edges = np.linspace(self.g.d0 + 0.5, self.g.far, 21)
        hist, _ = np.histogram(d, edges)
        hit = hist >= 2
        if line_fn is not None and self.visible is not None:
            dc = (edges[:-1] + edges[1:]) / 2
            u, v = self.g.metric_to_bev(dc, line_fn(dc))
            u = np.clip(u.astype(int), 0, self.g.bw - 1); v = np.clip(v.astype(int), 0, self.g.bh - 1)
            vis = self.visible[v, u] | hit
            return float(hit[vis].mean()) if vis.sum() >= 5 else 0.0
        return float(hit.mean())

    @staticmethod
    def _fit(d, lat):
        if len(d) < 20 or d.max() - d.min() < 4:
            return None
        deg = 2 if d.max() - d.min() > 15 else 1
        c = np.polyfit(d, lat, deg)
        res = np.abs(np.polyval(c, d) - lat)
        keep = res < 0.25
        if keep.sum() >= 15:
            c = np.polyfit(d[keep], lat[keep], deg)
        c = np.array([0.0, c[0], c[1]]) if deg == 1 else np.asarray(c, np.float64)
        c[0] = float(np.clip(c[0], -0.003, 0.003))
        c[1] = float(np.clip(c[1], -0.35, 0.35))
        return c

    def _sliding(self, d, lat, start):
        g = self.g
        mask = np.zeros(len(d), bool)
        edges = np.linspace(g.d0, g.d1, 14)
        cur, slope = start, 0.0
        for i in range(13):
            inw = (d >= edges[i]) & (d < edges[i + 1]) & (np.abs(lat - cur) < 0.45)
            if inw.sum() >= 4:
                mask |= inw
                new = float(np.median(lat[inw]))
                slope = 0.6 * slope + 0.4 * (new - cur)
                cur = new
            cur += slope * 0.3
        return mask

    # ---------------- main update ---------------------------------------------------
    def update(self, bev, block):
        g = self.g
        self.frame += 1
        self.switch = 0
        raw, y = self._binary(bev, block)
        b, rej, cross_d, stop_d = self._shape_filter(raw)
        self.last_binary, self.last_yellow, self.last_reject = b, y, rej
        self.crosswalk_d, self.stopline_d = cross_d, stop_d
        if (cross_d is not None and cross_d < 20.0) or (stop_d is not None and stop_d < 15.0):
            self.junction_until = self.frame + int(1.5 * self.fps)

        vs, us = np.nonzero(b)
        d, lat = g.bev_to_metric(vs.astype(np.float64), us.astype(np.float64))
        isy = y[vs, us] > 0
        free = np.ones(len(d), bool)

        # 1) update existing tracks
        for tr in sorted(self.tracks, key=lambda t: -t["hits"]):
            m = free & (np.abs(lat - pv(tr["c"], d)) < 0.4)
            c = self._fit(d[m], lat[m])
            if c is not None and abs(c[2] - tr["c"][2]) < 0.7:
                tr["c"] = 0.5 * c + 0.5 * tr["c"]
                tr["hits"] += 1; tr["miss"] = 0; tr["meas"] = True
                fn = (lambda cc: (lambda x: pv(cc, x)))(tr["c"])
                tr["cov"] = 0.85 * tr["cov"] + 0.15 * self._coverage(d[m], fn)
                tr["yel"] = 0.85 * tr["yel"] + 0.15 * float(isy[m].mean())
                free &= ~m
            else:
                tr["miss"] += 1; tr["meas"] = False

        # 2) new lines from remaining pixels (lateral histogram peaks)
        fd, fl, fy = d[free], lat[free], isy[free]
        if len(fd) > 30:
            near = fd < g.near + 0.6 * (g.far - g.near)
            bins = np.arange(-g.lat_half, g.lat_half + 0.1, 0.1)
            hist, _ = np.histogram(fl[near], bins)
            hist = np.convolve(hist, np.ones(3), "same")
            centers = (bins[:-1] + bins[1:]) / 2
            taken = [t["c"][2] for t in self.tracks]
            for idx in np.argsort(-hist)[:12]:
                if hist[idx] < 10:
                    break
                c0 = centers[idx]
                if any(abs(c0 - tk) < 1.6 for tk in taken):
                    continue
                m = self._sliding(fd, fl, c0)
                c = self._fit(fd[m], fl[m])
                if c is None or any(abs(c[2] - tk) < 1.0 for tk in taken):
                    continue
                fn = (lambda cc: (lambda x: pv(cc, x)))(c)
                self.tracks.append({"id": self.next_id, "c": c, "hits": 1, "miss": 0, "meas": True,
                                    "cov": self._coverage(fd[m], fn), "yel": float(fy[m].mean())})
                self.next_id += 1
                taken.append(c[2])

        # 3) merge duplicates, drop lost tracks
        self.tracks.sort(key=lambda t: -t["hits"])
        kept = []
        for tr in self.tracks:
            if any(abs(pv(tr["c"], self.d_ref) - pv(k["c"], self.d_ref)) < 0.9 for k in kept):
                continue
            kept.append(tr)
        junction = self.frame <= self.junction_until
        max_miss = int(4.0 * self.fps) if junction else int(0.6 * self.fps)
        self.tracks = [t for t in kept if t["miss"] <= max_miss and abs(t["c"][2]) < g.lat_half]

        conf_lines = [t for t in self.tracks if t["hits"] >= 4]
        if not any(t["meas"] for t in conf_lines):
            self.no_line_frames += 1
            if self.no_line_frames > int(0.5 * self.fps) and self.cl is not None:
                self.junction_until = max(self.junction_until, self.frame + 2)   # open junction / no paint
        else:
            self.no_line_frames = 0
        junction = self.frame <= self.junction_until
        self._select_ego(conf_lines, junction)
        return self.state()

    def _lat(self, t):
        return float(pv(t["c"], self.d_ref))

    def _select_ego(self, lines, junction):
        byid = {t["id"]: t for t in lines}
        lid, rid = self.ego
        L, R = byid.get(lid), byid.get(rid)
        hyst = 0.25
        if L is not None and self._lat(L) < -hyst:          # old left line now on our right -> moved LEFT
            if not junction:
                self.switch, self.crossed_type = 1, self.line_type(L["cov"])
            L = R = None
        elif R is not None and self._lat(R) > hyst:         # old right line now on our left -> moved RIGHT
            if not junction:
                self.switch, self.crossed_type = -1, self.line_type(R["cov"])
            L = R = None
        if L is None or R is None or (self._lat(L) - self._lat(R)) < 2.3:
            lefts = sorted([t for t in lines if self._lat(t) > -hyst], key=self._lat)
            rights = sorted([t for t in lines if self._lat(t) < hyst], key=self._lat, reverse=True)
            if L is not None and L in lefts:
                lefts = [L] + [t for t in lefts if t is not L]
            if R is not None and R in rights:
                rights = [R] + [t for t in rights if t is not R]
            best, bs = (None, None), 1e9
            for a in lefts[:3]:
                for b in rights[:3]:
                    if a is b:
                        continue
                    wd = self._lat(a) - self._lat(b)
                    if 2.3 < wd < 5.0:
                        s = abs(wd - self.w_est) + 0.3 * abs(self._lat(a) + self._lat(b)) - 0.002 * (a["hits"] + b["hits"])
                        if s < bs:
                            best, bs = (a, b), s
            if best[0] is not None:
                L, R = best
            else:
                L = lefts[0] if lefts and self._lat(lefts[0]) < 0.75 * self.w_est + 0.6 else None
                R = rights[0] if rights and self._lat(rights[0]) > -(0.75 * self.w_est + 0.6) else None
                if L is not None and R is not None and self._lat(L) - self._lat(R) < 2.3:
                    if L["hits"] >= R["hits"]: R = None
                    else: L = None
        self.ego = (L["id"] if L else None, R["id"] if R else None)

        if L is None and R is None:
            self.conf *= 0.85
            if self.cl is not None and not junction and self.no_line_frames > 2 * self.fps:
                self.cl = self.cr = None
            return
        if L is not None and R is not None:
            cl, cr = L["c"].copy(), R["c"].copy()
            wd = self._lat(L) - self._lat(R)
            if L["meas"] and R["meas"] and 2.4 < wd < 4.8:
                self.w_est = float(np.clip(0.97 * self.w_est + 0.03 * wd, 2.4, 4.8))
        elif L is not None:
            cl = L["c"].copy(); cr = cl - np.array([0, 0, self.w_est])
        else:
            cr = R["c"].copy(); cl = cr + np.array([0, 0, self.w_est])
        self.cl, self.cr = cl, cr
        q = 0.5 * (L["meas"] if L else 0) + 0.5 * (R["meas"] if R else 0)
        q *= 0.5 if junction else 1.0
        self.conf = 0.85 * self.conf + 0.15 * q

    @staticmethod
    def line_type(cov):
        return "solid" if cov >= 0.72 else ("dashed" if cov >= 0.12 else "none")

    def state(self):
        valid = self.cl is not None
        junction = self.frame <= self.junction_until
        lines = sorted([t for t in self.tracks if t["hits"] >= 4], key=self._lat, reverse=True)  # left -> right
        st = {"valid": valid, "cl": self.cl, "cr": self.cr, "w": self.w_est, "conf": float(self.conf),
              "switch": self.switch, "crossed_type": self.crossed_type, "far": self.g.far * 2.0,
              "junction": junction, "crosswalk_d": self.crosswalk_d, "stopline_d": self.stopline_d,
              "lines": [{"id": t["id"], "c": t["c"], "type": self.line_type(t["cov"]), "yellow": t["yel"] > 0.35,
                         "meas": t["meas"], "ego": t["id"] in self.ego} for t in lines]}
        byid = {t["id"]: t for t in lines}
        lid, rid = self.ego
        for key, tid in (("L", lid), ("R", rid)):
            t = byid.get(tid)
            st["type_" + key] = self.line_type(t["cov"]) if t else "none"
            st["yellow_" + key] = bool(t and t["yel"] > 0.35)
        if valid:
            cen = float(lane_center(st, self.d_ref))
            n_left = sum(1 for t in lines if self._lat(t) > cen + 0.5 * self.w_est + 0.6)
            n_right = sum(1 for t in lines if self._lat(t) < cen - 0.5 * self.w_est - 0.6)
            st["n_lanes"] = n_left + n_right + 1
            st["ego_lane_no"] = n_left + 1                 # counted from the left
        else:
            st["n_lanes"], st["ego_lane_no"] = 0, 0
        return st


def lane_center(lane, d):
    if not lane["valid"]:
        return np.zeros_like(np.asarray(d, np.float64))
    dm = lane["far"]
    return 0.5 * (pv_ext(lane["cl"], d, dm) + pv_ext(lane["cr"], d, dm))


def actor_lane_index(lane, d, lat):
    """0 = ego lane, +1 = first lane left, -1 = first lane right ... using real tracked lines."""
    w = lane["w"] if lane["valid"] else LANE_WIDTH_M
    rel = lat - float(lane_center(lane, d))
    if not lane["valid"] or abs(rel) > 3.0 * w:
        return int(round(rel / w)) if abs(rel) < 2.5 * w else None
    dm = lane["far"]
    L, R = float(pv_ext(lane["cl"], d, dm)), float(pv_ext(lane["cr"], d, dm))
    if R <= lat <= L:
        return 0
    others = [float(pv_ext(l["c"], d, dm)) for l in lane.get("lines", []) if not l["ego"]]
    if lat > L:
        return 1 + sum(1 for x in others if L + 1.5 < x < lat)
    return -(1 + sum(1 for x in others if lat < x < R - 1.5))


# =====================================================================
# Automatic camera calibration (vanishing point + lane-width scale)
# =====================================================================
def auto_calibrate(video, und=None, hfov_deg=None, cam_h=None):
    """Return (H, near, far, info). H maps image (u,v,1) -> ground (d, lat, 1)."""
    und = und or (lambda f: f)
    hfov_deg = hfov_deg or CAMERA_HFOV_DEG
    cam_h = cam_h or CAMERA_HEIGHT_M
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(30 * fps)
    span = min(total, int(AUTO_CALIB_SECONDS * fps))
    idxs = np.linspace(0, max(0, span - 1), 60).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(und(fr))
    cap.release()
    if not frames:
        raise RuntimeError("Cannot read frames for auto calibration")
    fh, fw = frames[0].shape[:2]
    segs = []
    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        y0 = int(0.45 * fh)
        e = cv2.Canny(cv2.GaussianBlur(gray[y0:int(0.97 * fh)], (5, 5), 0), 60, 160)
        ls = cv2.HoughLinesP(e, 1, np.pi / 180, 40, minLineLength=int(0.05 * fw), maxLineGap=25)
        if ls is None:
            continue
        for x1, y1, x2, y2 in ls[:, 0]:
            y1 += y0; y2 += y0
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            ang = min(ang, 180 - ang)
            if 18 < ang < 80:
                p1, p2 = np.array([x1, y1, 1.0]), np.array([x2, y2, 1.0])
                ln = np.cross(p1, p2); ln /= (np.hypot(ln[0], ln[1]) + 1e-9)
                segs.append((ln, math.hypot(x2 - x1, y2 - y1)))
    info = {"segments": len(segs)}
    vx, vy = fw / 2, 0.5 * fh
    if len(segs) >= 20:
        lines = np.array([s[0] for s in segs]); wts = np.array([s[1] for s in segs])
        rng = np.random.default_rng(0)
        best, bscore = None, -1
        tol = 0.012 * fw
        for _ in range(1500):
            i, j = rng.choice(len(lines), 2, replace=False)
            p = np.cross(lines[i], lines[j])
            if abs(p[2]) < 1e-9:
                continue
            p = p / p[2]
            if not (0.1 * fw < p[0] < 0.9 * fw and 0.15 * fh < p[1] < 0.75 * fh):
                continue
            score = wts[np.abs(lines @ p) < tol].sum()
            if score > bscore:
                best, bscore = p, score
        if best is not None:
            inl = np.abs(lines @ best) < tol
            A = lines[inl, :2] * wts[inl, None]; bvec = -lines[inl, 2] * wts[inl]
            sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
            if 0.1 * fw < sol[0] < 0.9 * fw and 0.15 * fh < sol[1] < 0.75 * fh:
                vx, vy = float(sol[0]), float(sol[1])
                info["vp_inlier_weight"] = float(bscore / wts.sum())
    info["vanishing_point"] = [vx, vy]
    f = float(np.asarray(CAMERA_K)[0, 0]) if CAMERA_K is not None else (fw / 2) / math.tan(math.radians(hfov_deg) / 2)

    def build(h):
        H = np.array([[0, 0, h * f], [-h, 0, h * vx], [0, 1, -vy]], np.float64)
        d_bottom = h * f / max(1.0, fh - vy)
        near = max(3.0, d_bottom + 1.5)
        return H, near, near + 24.0
    H, near, far = build(cam_h)
    if AUTO_SCALE_FROM_LANE_WIDTH:
        geo = GroundGeometry(fw, fh, None, near, far, LANE_WIDTH_M, H=H)
        ld = LaneDetector(geo, LANE_WIDTH_M, fps=10)
        widths = []
        for fr in frames:
            bev = cv2.warpPerspective(fr, geo.M, (geo.bw, geo.bh))
            ld.update(bev, np.zeros((geo.bh, geo.bw), np.uint8))
            byid = {t["id"]: t for t in ld.tracks}
            a, b = ld.ego
            if a in byid and b in byid and byid[a]["meas"] and byid[b]["meas"]:
                widths.append(ld._lat(byid[a]) - ld._lat(byid[b]))
        info["width_samples"] = len(widths)
        if len(widths) >= 12:
            wm = float(np.median(widths))
            k = LANE_WIDTH_M / wm
            if 0.55 < k < 1.8:
                cam_h *= k
                H, near, far = build(cam_h)
                info["measured_width_before_scale"] = wm
    info.update({"focal_px": f, "camera_height_m": cam_h, "near_m": near, "far_m": far})
    LOG.info("Auto calibration: %s", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}))
    return H, near, far, info


# =====================================================================
# Ego speed estimate (template matching of BEV road-texture gradients)
# =====================================================================
class EgoSpeedEstimator:
    def __init__(self, geo: GroundGeometry):
        self.g = geo
        self.prev = None
        self.hist = deque(maxlen=9)
        self.speed = None

    def update(self, bev, block, dt):
        if EGO_SPEED_KPH is not None:
            self.speed = EGO_SPEED_KPH / 3.6
            return self.speed
        g = self.g
        v0, v1 = int((g.d1 - g.far) * g.my), int((g.d1 - g.near) * g.my)
        u0, u1 = int(g.bw * 0.25), int(g.bw * 0.75)
        gray = cv2.cvtColor(bev[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = (block[v0:v1, u0:u1] > 0) | (gray <= 6)     # vehicles + area outside the camera image
        gray = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=3))
        mm = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if (~mm).sum() < 0.15 * mm.size:
            self.prev = None
            return self.speed
        gray -= float(gray[~mm].mean())
        gray[mm] = 0.0
        if self.prev is not None and self.prev.shape == gray.shape and dt > 0:
            h, w = gray.shape
            up, down, side = 4, min(40, h // 3), 8
            tpl = self.prev[up:h - down, side:w - side]
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, (bx, by) = cv2.minMaxLoc(res)
            if score > 0.15:
                dy = float(by - up)
                if 0 < by < res.shape[0] - 1:
                    a, b, c = res[by - 1, bx], res[by, bx], res[by + 1, bx]
                    den = a - 2 * b + c
                    if abs(den) > 1e-9:
                        dy += 0.5 * (a - c) / den
                spd = max(0.0, dy) / g.my / dt
                if spd < 60:
                    self.hist.append(spd)
        self.prev = gray
        if len(self.hist) >= 3:
            med = float(np.median(self.hist))
            self.speed = med if self.speed is None else 0.8 * self.speed + 0.2 * med
        return self.speed


# =====================================================================
# Events
# =====================================================================
class EventLog:
    def __init__(self):
        self.items, self.last = [], {}

    def add(self, t, frame, typ, msg, sev="info", key=None, cooldown=0.0):
        if key is not None and key in self.last and t - self.last[key] < cooldown:
            return
        if key is not None:
            self.last[key] = t
        self.items.append({"time_s": round(t, 2), "frame": frame, "type": typ, "severity": sev, "message": msg})
        LOG.info("[%6.2fs] %s", t, msg)


# =====================================================================
# Ego lane-change monitor
# =====================================================================
class LaneChangeMonitor:
    def __init__(self, fps):
        self.state, self.dir, self.crossed = "KEEP", 0, False
        self.t_start = self.t_cross = 0.0
        self.hist = deque(maxlen=max(5, int(fps * 0.7)))
        self.off = self.vel = 0.0
        self.lane_index = 0
        self.count = {"LEFT": 0, "RIGHT": 0}

    def update(self, t, lane, events, frame):
        if not lane["valid"]:
            return
        if lane.get("junction"):
            self.hist.clear(); self.vel = 0.0
            if self.state != "KEEP" and not self.crossed:
                self.state, self.dir = "KEEP", 0
            return
        w = lane["w"]
        off = -(lane["cl"][2] + lane["cr"][2]) / 2       # +ve = ego left of lane centre
        sw = lane["switch"]
        if sw:
            self.hist = deque([(tt, o - sw * w) for tt, o in self.hist], maxlen=self.hist.maxlen)
        self.hist.append((t, off))
        if len(self.hist) >= 4:
            ts = np.array([h[0] for h in self.hist]); os_ = np.array([h[1] for h in self.hist])
            if np.ptp(ts) > 1e-6:
                self.vel = float(np.polyfit(ts, os_, 1)[0])
        self.off = off
        if sw:
            name = "LEFT" if sw > 0 else "RIGHT"
            self.lane_index += sw
            self.count[name] += 1
            solid = lane["crossed_type"] == "solid"
            events.add(t, frame, "ego_lane_change", f"Ego lane change {name} completed"
                       + (" - crossed SOLID line" if solid else ""), "warn" if solid else "info")
            if self.state == "KEEP":
                self.t_start = t
            self.state, self.dir, self.crossed, self.t_cross = "CHANGING_" + name, sw, True, t
            return
        if self.state == "KEEP":
            if lane["conf"] < 0.45 or len(self.hist) < self.hist.maxlen:
                pass
            elif self.vel > 0.3 and off > 0.25 * w:
                self._start(t, 1, events, frame)
            elif self.vel < -0.3 and off < -0.25 * w:
                self._start(t, -1, events, frame)
        else:
            if self.crossed and ((abs(off) < 0.18 * w and abs(self.vel) < 0.3) or t - self.t_cross > 4):
                self.state, self.dir, self.crossed = "KEEP", 0, False
            elif not self.crossed and abs(off) < 0.1 * w:
                events.add(t, frame, "ego_lane_change", "Ego lane change aborted", "info")
                self.state, self.dir = "KEEP", 0
            elif not self.crossed and t - self.t_start > 8:
                self.state, self.dir = "KEEP", 0

    def _start(self, t, s, events, frame):
        name = "LEFT" if s > 0 else "RIGHT"
        self.state, self.dir, self.crossed, self.t_start = "CHANGING_" + name, s, False, t
        events.add(t, frame, "ego_lane_change", f"Ego lane change {name} initiated", "info")


# =====================================================================
# Detection + tracking (YOLO + ByteTrack)
# =====================================================================
class YoloTracker:
    def __init__(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install detector: python -m pip install ultralytics lap") from exc
        try:
            self.model = YOLO(MODEL_WEIGHTS)
        except Exception as exc:
            LOG.warning("Cannot load %s (%s); using %s", MODEL_WEIGHTS, exc, FALLBACK_WEIGHTS)
            self.model = YOLO(FALLBACK_WEIGHTS)
        names = self.model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        self.classes = None
        if KEEP_CLASSES:
            ids = [i for i, n in self.names.items() if n.lower() in KEEP_CLASSES]
            self.classes = ids or None
        self.device, self.half = "cpu", False
        try:
            import torch
            if torch.cuda.is_available():
                self.device, self.half = 0, True
        except Exception:
            pass
        LOG.info("Detector classes: %s | device: %s", [self.names[i] for i in (self.classes or self.names)], self.device)

    def detect(self, frame):
        r = self.model.track(frame, persist=True, tracker=TRACKER_CFG, conf=DET_CONF, iou=DET_IOU,
                             imgsz=IMG_SIZE, classes=self.classes, device=self.device, half=self.half,
                             verbose=False)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(xyxy)
        for b, c, k, i in zip(xyxy, conf, cls, ids):
            out.append({"id": None if i is None else int(i), "cls": self.names[int(k)], "conf": float(c),
                        "bbox": tuple(float(v) for v in b)})
        return out


def traffic_light_state(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size < 30:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lit = (s > 90) & (v > 140)
    counts = {"red": (((h < 10) | (h > 160)) & lit).sum(), "yellow": ((h >= 15) & (h <= 35) & lit).sum(),
              "green": ((h >= 40) & (h <= 95) & lit).sum()}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0.02 * h.size else "unknown"


class Track:
    def __init__(self, tid):
        self.id = tid
        self.votes = Counter()
        self.light_votes = Counter()
        self.hist = deque(maxlen=24)
        self.lane_hist = deque(maxlen=6)
        self.d = self.lat = None
        self.vd = self.vlat = 0.0
        self.bbox, self.conf = None, 0.0
        self.lane = None
        self.missed = 0
        self.flag_until, self.flag_text = -1.0, ""
        self.intent = 0
        self.ttc = None

    @property
    def cls(self):
        return self.votes.most_common(1)[0][0] if self.votes else "car"

    @property
    def kind(self):
        return kind_of(self.cls)

    @property
    def light(self):
        return self.light_votes.most_common(1)[0][0] if self.light_votes else "unknown"


class TrackManager:
    def __init__(self, fps):
        self.tracks = {}
        self.fps = fps
        self.lead_id = None

    def update(self, dets, frame, geo, lane, t, fidx, dt, events):
        seen = set()
        sw = lane.get("switch", 0)
        if sw:  # ego changed lane -> re-index actor lanes so no false cut-in/out is reported
            for tr in self.tracks.values():
                if tr.lane is not None:
                    tr.lane -= sw
                tr.lane_hist = deque([None if v is None else v - sw for v in tr.lane_hist], maxlen=6)
        for det in dets:
            if det["id"] is None:
                continue
            tr = self.tracks.get(det["id"])
            if tr is None:
                tr = self.tracks[det["id"]] = Track(det["id"])
            seen.add(tr.id)
            tr.votes[det["cls"]] += det["conf"]
            tr.bbox, tr.conf, tr.missed = det["bbox"], det["conf"], 0
            if tr.kind == "light":
                tr.light_votes[traffic_light_state(frame, det["bbox"])] += 1
                if sum(tr.light_votes.values()) > 15:
                    tr.light_votes = Counter(dict(tr.light_votes.most_common(2)))
                continue
            x1, y1, x2, y2 = det["bbox"]
            g = geo.img_to_ground((x1 + x2) / 2, y2)
            if g is None:
                continue
            d, lat = g
            if tr.d is None:
                tr.d, tr.lat = d, lat
            else:
                pd, pl = tr.d + tr.vd * dt, tr.lat + tr.vlat * dt
                a = 0.45 if abs(d - pd) < max(4.0, 0.25 * pd) else 0.15
                tr.d, tr.lat = a * d + (1 - a) * pd, 0.4 * lat + 0.6 * pl
            tr.hist.append((t, tr.d, tr.lat))
            recent = [h for h in tr.hist if t - h[0] <= 0.7]
            if len(recent) >= 4:
                ts = np.array([h[0] for h in recent])
                if np.ptp(ts) > 1e-6:
                    tr.vd = float(np.polyfit(ts, [h[1] for h in recent], 1)[0])
                    tr.vlat = float(np.polyfit(ts, [h[2] for h in recent], 1)[0])
            self._lane_logic(tr, lane, t, fidx, events)

        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tid in seen:
                continue
            tr.missed += 1
            if tr.d is not None:
                tr.d += tr.vd * dt; tr.lat += tr.vlat * dt
            if tr.missed > 8:
                del self.tracks[tid]

        self.lead_id, best = None, 1e9
        for tr in self.tracks.values():
            tr.ttc = None
            if tr.d is None or tr.missed > 0:
                continue
            if tr.vd < -0.3 and tr.d > 0:
                tr.ttc = tr.d / -tr.vd
            if tr.kind in VEHICLE_KINDS and tr.lane == 0 and 0 < tr.d < best:
                best, self.lead_id = tr.d, tr.id
            if tr.kind == "person" and tr.lane == 0 and tr.d < 30:
                events.add(t, fidx, "vru", f"Pedestrian #{tr.id} in ego path at {tr.d:.1f} m", "warn",
                           key=("ped", tr.id), cooldown=5)
            if tr.ttc is not None and tr.ttc < 2.5 and abs(tr.lat - float(lane_center(lane, tr.d))) < 2.5:
                events.add(t, fidx, "ttc", f"Critical TTC {tr.ttc:.1f} s with {tr.cls} #{tr.id}", "crit",
                           key=("ttc", tr.id), cooldown=4)
            if tr.kind == "light" and tr.light == "red":
                events.add(t, fidx, "traffic_light", "Red traffic light ahead", "warn", key="redlight", cooldown=8)

    def _lane_logic(self, tr, lane, t, fidx, events):
        w = lane["w"] if lane["valid"] else LANE_WIDTH_M
        rel = tr.lat - float(lane_center(lane, tr.d))
        idx = actor_lane_index(lane, tr.d, tr.lat)
        tr.lane_hist.append(idx)
        vals = [v for v in tr.lane_hist if v is not None]
        if not vals:
            return
        mode, cnt = Counter(vals).most_common(1)[0]
        prev = tr.lane
        if cnt >= 4 or prev is None:
            tr.lane = mode
        boundary = (tr.lane + 0.5 * np.sign(tr.vlat)) * w if tr.vlat != 0 else 1e9
        tr.intent = int(np.sign(tr.vlat)) if abs(tr.vlat) > 0.35 and abs(rel - boundary) < 0.9 else 0
        if prev is not None and tr.lane != prev and tr.kind in VEHICLE_KINDS and not lane.get("junction"):
            side = "LEFT" if prev > 0 else "RIGHT"
            if tr.lane == 0:
                msg, sev, flag = f"{tr.cls.title()} #{tr.id} CUT-IN from {side}", "warn", "CUT-IN"
            elif prev == 0:
                msg, sev, flag = f"{tr.cls.title()} #{tr.id} cut-out to {'LEFT' if tr.lane > 0 else 'RIGHT'}", "info", "CUT-OUT"
            else:
                msg, sev, flag = f"{tr.cls.title()} #{tr.id} lane change", "info", "LANE CHG"
            events.add(t, fidx, "actor_lane_change", msg, sev, key=("lc", tr.id, flag), cooldown=2)
            tr.flag_until, tr.flag_text = t + 2.0, flag


# =====================================================================
# 3-D digital-twin renderer (OpenCV, super-sampled)
# =====================================================================
def _norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


PALETTES = {
    "light": dict(sky_top=(232, 230, 228), sky_bot=(246, 246, 246), ground=(240, 240, 241), road=(222, 222, 224),
                  shoulder=(212, 212, 214), line=(150, 150, 155), yellow=(40, 185, 235), faint=(205, 205, 208),
                  other=(206, 206, 210), lead=(240, 132, 58), cut=(30, 140, 255), crit=(70, 70, 235),
                  ego=(58, 56, 60), glass=(95, 85, 78), person=(105, 105, 110), path=(245, 160, 70),
                  chevron=(225, 120, 40), edge=(120, 120, 125), text=(40, 40, 45), tag_bg=(255, 255, 255)),
    "dark": dict(sky_top=(30, 22, 16), sky_bot=(48, 38, 30), ground=(40, 34, 28), road=(60, 54, 50),
                 shoulder=(50, 45, 42), line=(225, 225, 225), yellow=(40, 195, 245), faint=(90, 85, 80),
                 other=(185, 185, 190), lead=(250, 150, 60), cut=(30, 150, 255), crit=(70, 70, 240),
                 ego=(235, 235, 240), glass=(70, 55, 45), person=(200, 200, 205), path=(255, 170, 60),
                 chevron=(255, 200, 120), edge=(40, 40, 40), text=(240, 240, 240), tag_bg=(40, 32, 26)),
}


def prism(x0, x1, y0, y1, z0, z1, sf=0.0, sb=0.0, ss=0.0, kind="body"):
    """Local box with top face inset: sf=front, sb=back, ss=sides."""
    bot = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    top = [(x0 + sb, y0 + ss, z1), (x1 - sf, y0 + ss, z1), (x1 - sf, y1 - ss, z1), (x0 + sb, y1 - ss, z1)]
    return {"bot": bot, "top": top, "kind": kind}


def _wheels(L, W, inset, axles):
    xs = [L / 2 - inset, -L / 2 + inset] if axles == 2 else [L / 2 - inset, -L / 2 + inset, -L / 2 + inset + 1.4]
    out = []
    for x in xs:
        for y in (-W / 2 - 0.02, W / 2 - 0.26):
            out.append(prism(x - 0.34, x + 0.34, y, y + 0.28, 0.0, 0.66, kind="wheel"))
    return out


def model_parts(kind, L, W):
    h, w = L / 2, W / 2
    if kind == "truck":
        return [prism(-h, h, -w, w, 0.45, 0.95, kind="chassis"),
                prism(-h, h - 2.3, -w, w, 0.95, 3.4, 0.02, 0.02, 0.02),
                prism(h - 2.2, h, -w + 0.05, w - 0.05, 0.95, 2.9, 0.35, 0.0, 0.08, kind="cab"),
                prism(h - 0.25, h - 0.05, -w + 0.2, w - 0.2, 1.9, 2.7, kind="glass")] + _wheels(L, W, 0.95, 3)
    if kind == "bus":
        return [prism(-h, h, -w, w, 0.35, 1.5),
                prism(-h, h, -w, w, 1.5, 2.6, kind="glass"),
                prism(-h, h, -w, w, 2.6, 3.1, 0.1, 0.1, 0.08)] + _wheels(L, W, 0.95, 2)
    if kind in ("motorcycle", "bicycle"):
        bw = 0.22 if kind == "motorcycle" else 0.12
        return [prism(-0.8, 0.8, -bw, bw, 0.3, 0.85 if kind == "motorcycle" else 0.7, 0.2, 0.2),
                prism(0.55, 0.95, -0.06, 0.06, 0.0, 0.62, kind="wheel"),
                prism(-0.95, -0.55, -0.06, 0.06, 0.0, 0.62, kind="wheel"),
                prism(-0.35, 0.05, -0.22, 0.22, 0.85, 1.45, 0.05, 0.05, 0.04, kind="rider"),
                prism(-0.25, 0.0, -0.12, 0.12, 1.47, 1.75, kind="head")]
    if kind == "person":
        return [prism(-0.12, 0.12, -0.18, 0.18, 0.0, 0.85, kind="rider"),
                prism(-0.14, 0.14, -0.24, 0.24, 0.85, 1.45, 0.03, 0.03, 0.04, kind="rider"),
                prism(-0.11, 0.11, -0.1, 0.1, 1.5, 1.75, kind="head")]
    if kind == "tractor":
        return [prism(-0.6, 1.7, -0.45, 0.45, 0.6, 1.35, 0.15, 0.0, 0.05),
                prism(-1.6, -0.4, -0.6, 0.6, 1.2, 2.6, 0.1, 0.1, 0.05, kind="glass"),
                prism(-1.65, -0.35, -0.65, 0.65, 2.6, 2.75),
                prism(-1.6, -0.6, -1.0, -0.6, 0.0, 1.45, kind="wheel"), prism(-1.6, -0.6, 0.6, 1.0, 0.0, 1.45, kind="wheel"),
                prism(1.0, 1.6, -0.85, -0.6, 0.0, 0.85, kind="wheel"), prism(1.0, 1.6, 0.6, 0.85, 0.0, 0.85, kind="wheel")]
    if kind == "auto":
        return [prism(-h, h, -w, w, 0.25, 0.9, 0.1, 0.0, 0.0),
                prism(-h, h - 0.3, -w + 0.05, w - 0.05, 0.9, 1.75, 0.35, 0.15, 0.1, kind="glass"),
                prism(-h - 0.05, h - 0.2, -w, w, 1.75, 1.85)] + _wheels(L, W, 0.5, 2)
    return [prism(-h, h, -w, w, 0.28, 0.92, 0.12, 0.08, 0.05),
            prism(-h + 0.5, h - 1.3, -w + 0.12, w - 0.12, 0.92, 1.45, 0.55, 0.35, 0.12, kind="glass"),
            prism(-h - 0.02, -h + 0.02, -w + 0.1, -w + 0.45, 0.68, 0.8, kind="tail"),
            prism(-h - 0.02, -h + 0.02, w - 0.45, w - 0.1, 0.68, 0.8, kind="tail")] + _wheels(L, W, 0.85, 2)


DIMS = {"car": (4.5, 1.85), "truck": (8.5, 2.5), "bus": (11.0, 2.55), "motorcycle": (2.0, 0.8),
        "bicycle": (1.8, 0.6), "person": (0.5, 0.6), "tractor": (3.6, 1.9), "auto": (2.7, 1.4)}


class TwinRenderer:
    def __init__(self, w, h, ss=2, theme="light"):
        self.w, self.h, self.ss = w, h, max(1, int(ss))
        self.W, self.H = w * self.ss, h * self.ss
        self.p = PALETTES.get(theme, PALETTES["light"])
        self.f = 1.05 * self.W
        self.cx, self.cy = self.W / 2, 0.36 * self.H
        self.cam = np.array([-15.0, 0.0, 6.2])
        fwd = _norm(np.array([32.0, 0, 0]) - self.cam)
        right = _norm(np.cross(fwd, [0, 0, 1.0]))
        up = np.cross(right, fwd)
        self.R = np.stack([right, up, fwd])
        self.light = _norm([-0.35, 0.45, 1.0])
        self.horizon = int(self.proj(np.array([[5000.0, 0, 0]]))[0][0, 1])
        bg = np.zeros((self.H, self.W, 3), np.uint8)
        a = np.linspace(0, 1, max(1, self.horizon))[:, None, None]
        bg[:self.horizon] = (np.array(self.p["sky_top"]) * (1 - a) + np.array(self.p["sky_bot"]) * a).astype(np.uint8)
        bg[self.horizon:] = self.p["ground"]
        self.bg = bg
        r0, r1 = self.horizon, min(self.H, self.horizon + int(0.22 * self.H))
        self.fog_rows = (r0, r1)
        fa = np.linspace(1.0, 0.0, max(0, r1 - r0), dtype=np.float32) ** 1.6
        self.fog_w2 = np.ascontiguousarray(np.repeat(fa[:, None], self.W, 1))
        self.fog_w1 = 1.0 - self.fog_w2
        self.fog_bg = np.full((max(0, r1 - r0), self.W, 3), self.p["sky_bot"], np.uint8)
        self.odometer = 0.0
        self.cross = None
        self._src_w = 1920

    def proj(self, P):
        Q = (np.asarray(P, np.float64) - self.cam) @ self.R.T
        z = Q[:, 2]
        zz = np.maximum(z, 1e-3)
        return np.stack([self.cx + self.f * Q[:, 0] / zz, self.cy - self.f * Q[:, 1] / zz], 1), z

    def ground_poly(self, img, d, lat_a, lat_b, color):
        P = np.concatenate([np.stack([d, lat_a, np.zeros_like(d)], 1),
                            np.stack([d[::-1], lat_b[::-1], np.zeros_like(d)], 1)])
        uv, z = self.proj(P)
        if (z > 0.5).sum() < 3:
            return
        pts = np.round(uv[z > 0.5] * 4).astype(np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA, shift=2)

    def ribbon(self, img, d, lat, hw, color):
        self.ground_poly(img, d, lat + hw, lat - hw, color)

    def draw_road(self, img, lane, lc):
        p = self.p
        ds = np.linspace(-9.5, 120, 100)
        dm = lane["far"]
        if lane["valid"]:
            L, R, w = pv_ext(lane["cl"], ds, dm), pv_ext(lane["cr"], ds, dm), lane["w"]
        else:
            w = LANE_WIDTH_M
            L, R = np.full_like(ds, w / 2), np.full_like(ds, -w / 2)
        others = []
        if lane["valid"]:
            for ln in lane.get("lines", []):
                if not ln["ego"]:
                    others.append((pv_ext(ln["c"], ds, dm), ln))
        lat_all = [L, R] + [o[0] for o in others]
        left_edge = np.max(np.vstack(lat_all), 0) + 1.2
        right_edge = np.min(np.vstack(lat_all), 0) - 1.2
        self.ground_poly(img, ds, left_edge + 1.5, right_edge - 1.5, p["shoulder"])
        self.ground_poly(img, ds, left_edge, right_edge, p["road"])
        if SHOW_PATH_PLANNING and lc.state != "KEEP" and lane["valid"]:
            if not lc.crossed:
                a, b = (L + w, L) if lc.dir > 0 else (R, R - w)
            else:
                a, b = L, R
            ov = img.copy()
            self.ground_poly(ov, ds, a, b, p["path"])
            cv2.addWeighted(ov, 0.22, img, 0.78, 0, img)
        lines = [(lane.get("type_L", "dashed"), lane.get("yellow_L"), L), (lane.get("type_R", "dashed"), lane.get("yellow_R"), R)]
        for lat, ln in others:
            lines.append((ln["type"], ln["yellow"], lat))
        junction = lane.get("junction")
        for typ, yel, lat in lines:
            col = p["yellow"] if yel else p["line"]
            if typ == "none" or not lane["valid"]:
                col, typ = p["faint"], "solid"
            if junction:
                col = tuple(int(0.5 * a + 0.5 * b) for a, b in zip(col, p["road"]))
            self.marking(img, ds, lat, typ, col)
        # crosswalk (coasted with ego speed)
        if lane.get("crosswalk_d") is not None:
            self.cross = [lane["crosswalk_d"], 0.0]
        if self.cross is not None:
            cd = self.cross[0] - self.cross[1]
            if cd < -12:
                self.cross = None
            else:
                lo, hi = float(np.interp(cd, ds, right_edge)) + 0.6, float(np.interp(cd, ds, left_edge)) - 0.6
                for y0 in np.arange(lo, hi - 0.5, 1.0):
                    self.ground_poly(img, np.array([cd - 1.5, cd + 1.5]), np.array([y0 + 0.5, y0 + 0.5]),
                                     np.array([y0, y0]), p["line"])
        self.ground_poly(img, ds, left_edge + 0.08, left_edge - 0.08, p["edge"])
        self.ground_poly(img, ds, right_edge + 0.08, right_edge - 0.08, p["edge"])
        return ds, L, R, w

    def marking(self, img, ds, lat, typ, color, hw=0.08):
        if typ == "solid":
            self.ribbon(img, ds, lat, hw, color)
            return
        period, dash = 9.0, 3.0
        start = -9.5 - (self.odometer % period)
        while start < 120:
            a, b = max(start, -9.5), min(start + dash, 120)
            if b > a:
                seg = np.linspace(a, b, 5)
                self.ribbon(img, seg, np.interp(seg, ds, lat), hw, color)
            start += period

    def draw_path(self, img, lane, lc, lead_d):
        if not SHOW_PATH_PLANNING:
            return
        p = self.p
        end = min(60.0, lead_d - 1.0) if lead_d is not None else 60.0
        if end < 2:
            return
        ds = np.linspace(0.6, end, 60)
        center = lane_center(lane, ds)
        w = lane["w"] if lane["valid"] else LANE_WIDTH_M
        target, blend = center, 12.0
        if lc.state != "KEEP" and not lc.crossed:
            target, blend = center + lc.dir * w, 22.0
        elif lc.state != "KEEP":
            blend = 18.0
        s = np.clip(ds / blend, 0, 1); s = s * s * (3 - 2 * s)
        path = s * target
        ov = img.copy()
        self.ribbon(ov, ds, path, 0.95, p["path"])
        cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
        for i in range(4):
            dc = 1.6 + i * 1.5
            if dc > end - 1:
                break
            off = float(np.interp(dc, ds, path))
            V = [(dc, 0.7), (dc + 0.6, 0.0), (dc, -0.7), (dc - 0.35, -0.7), (dc + 0.22, 0.0), (dc - 0.35, 0.7)]
            P = np.array([[a, b + off, 0.01] for a, b in V])
            uv, z = self.proj(P)
            cv2.fillPoly(img, [np.round(uv * 4).astype(np.int32)], p["chevron"], cv2.LINE_AA, shift=2)

    def object_faces(self, kind, x, y, yaw, base_color, faces, shadows):
        L, W = DIMS.get(kind, DIMS["car"])
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T = np.array([x, y, 0.0])
        p = self.p
        fp = np.array([[-L / 2 - 0.2, -W / 2 - 0.2, 0], [L / 2 + 0.2, -W / 2 - 0.2, 0],
                       [L / 2 + 0.2, W / 2 + 0.2, 0], [-L / 2 - 0.2, W / 2 + 0.2, 0]]) @ Rz.T + T + [0.25, -0.35, 0]
        shadows.append(fp)
        for part in model_parts(kind, L, W):
            k = part["kind"]
            col = {"glass": p["glass"], "wheel": (35, 35, 38), "tail": (40, 40, 210), "chassis": (70, 70, 75),
                   "head": (150, 175, 205), "rider": p["person"]}.get(k, base_color)
            if kind == "person" and k == "rider":
                col = base_color
            bot = np.array(part["bot"]) @ Rz.T + T
            top = np.array(part["top"]) @ Rz.T + T
            cen = np.vstack([bot, top]).mean(0)
            quads = [top, [bot[0], bot[1], top[1], top[0]], [bot[1], bot[2], top[2], top[1]],
                     [bot[2], bot[3], top[3], top[2]], [bot[3], bot[0], top[0], top[3]]]
            for qi, q in enumerate(quads):
                q = np.asarray(q)
                fc = q.mean(0)
                n = _norm(np.cross(q[1] - q[0], q[2] - q[0]))
                if np.dot(n, fc - cen) < 0:
                    n = -n
                if np.dot(n, self.cam - fc) <= 0:
                    continue
                shade = 0.58 + 0.42 * max(0.0, float(np.dot(n, self.light)))
                if qi == 0:
                    shade = min(1.08, shade + 0.08)
                fcol = tuple(int(min(255, v * shade)) for v in col)
                uv, z = self.proj(q)
                if (z < 0.5).any():
                    continue
                faces.append((float(np.linalg.norm(fc - self.cam)), uv, fcol))

    def render(self, lane, lc, tracks, lead_id, speed, t, dt):
        p = self.p
        if speed:
            self.odometer += speed * dt
            if self.cross is not None:
                self.cross[1] += speed * dt
        img = self.bg.copy()
        ds, L, R, w = self.draw_road(img, lane, lc)
        lead = tracks.get(lead_id) if lead_id is not None else None
        self.draw_path(img, lane, lc, lead.d if lead is not None and lead.d is not None else None)

        faces, shadows, tags, lights = [], [], [], []
        for tr in tracks.values():
            if tr.kind == "light":
                if tr.missed <= 3 and tr.bbox is not None:
                    lights.append(tr)
                continue
            if tr.d is None or tr.missed > 5 or not (-12 < tr.d < 115) or abs(tr.lat) > 18:
                continue
            kind = tr.kind if tr.kind in DIMS else "car"
            cx = tr.d + (DIMS[kind][0] / 2 if kind != "person" else 0)
            slope = float(np.interp(cx, ds, np.gradient(lane_center(lane, ds), ds))) if lane["valid"] else 0.0
            yaw = math.atan(slope) + float(np.clip(math.atan2(tr.vlat, max(5.0, (speed or 10) + tr.vd)), -0.25, 0.25))
            if kind == "person":
                col = p["crit"] if tr.lane == 0 and tr.d < 30 else p["person"]
            elif tr.ttc is not None and tr.ttc < 2.5 and tr.lane == 0:
                col = p["crit"]
            elif t < tr.flag_until or tr.intent:
                col = p["cut"]
            elif tr.id == lead_id:
                col = p["lead"]
            else:
                col = p["other"]
            self.object_faces(kind, cx, tr.lat, yaw, col, faces, shadows)
            if SHOW_LABELS and tr.d < 75:
                uv, z = self.proj(np.array([[cx, tr.lat, (3.6 if kind in ("truck", "bus") else 2.1)]]))
                label = f"{tr.cls.title()} #{tr.id}  {tr.d:.0f} m"
                if t < tr.flag_until: label += "  " + tr.flag_text
                elif tr.ttc is not None and tr.ttc < 4: label += f"  TTC {tr.ttc:.1f}s"
                tags.append((uv[0], label, col, z[0]))

        ego_yaw = float(np.clip(math.atan2(lc.vel, max(5.0, speed or 10)), -0.14, 0.14)) if lc.state != "KEEP" else 0.0
        self.object_faces("car", -1.9, 0.0, ego_yaw, p["ego"], faces, shadows)

        sm = np.zeros((self.H // 4, self.W // 4), np.uint8)
        for fp in shadows:
            uv, z = self.proj(fp)
            if (z > 0.5).all():
                cv2.fillPoly(sm, [np.round(uv).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        if sm.any():
            sm = cv2.GaussianBlur(sm, (0, 0), 3)
            r0 = max(0, self.horizon)
            m = cv2.resize(sm, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[r0:].astype(np.float32) * (0.28 / 255.0)
            roi = np.ascontiguousarray(img[r0:])
            img[r0:] = cv2.blendLinear(roi, np.zeros_like(roi), 1.0 - m, m)

        for _, uv, col in sorted(faces, key=lambda f: -f[0]):
            pts = np.round(uv * 4).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA, shift=2)
            cv2.polylines(img, [pts], True, tuple(int(c * 0.75) for c in col), max(1, self.ss // 2), cv2.LINE_AA, shift=2)

        for tr in lights:
            x1, _, x2, _ = tr.bbox
            lat = float(np.clip(-((x1 + x2) / 2 / max(1, self._src_w) - 0.5) * 26, -11, 11))
            self.traffic_light(img, 55.0, lat, tr.light)

        r0, r1 = self.fog_rows
        if r1 > r0:
            img[r0:r1] = cv2.blendLinear(np.ascontiguousarray(img[r0:r1]), self.fog_bg, self.fog_w1, self.fog_w2)

        out = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        placed = []
        for u, label, col, z in sorted(tags, key=lambda q: -q[0][1]):
            if z <= 0.5:
                continue
            x, y = u[0] / self.ss, u[1] / self.ss
            tw = text_w(label, 0.45) + 16
            for _ in range(4):
                if not any(abs(x - px) < (tw + pw) / 2 + 4 and abs(y - py) < 25 for px, py, pw in placed):
                    break
                y -= 26
            placed.append((x, y, tw))
            if 0 < x < self.w and 30 < y < self.h:
                rrect(out, x - tw / 2, y - 22, tw, 22, 8, p["tag_bg"])
                cv2.circle(out, (int(x - tw / 2 + 9), int(y - 11)), 4, col, -1, cv2.LINE_AA)
                put(out, label, x - tw / 2 + 16, y - 6, 0.45, p["text"])
        return out

    def traffic_light(self, img, d, lat, state):
        pole = np.array([[d, lat, 0], [d, lat, 5.2]])
        uv, z = self.proj(pole)
        if (z < 0.5).any():
            return
        cv2.line(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), (90, 90, 95), max(2, 2 * self.ss), cv2.LINE_AA)
        box = np.array([[d, lat - 0.25, 4.3], [d, lat + 0.25, 4.3], [d, lat + 0.25, 5.6], [d, lat - 0.25, 5.6]])
        uv, _ = self.proj(box)
        cv2.fillPoly(img, [np.int32(uv)], (60, 60, 65), cv2.LINE_AA)
        colors = {"red": (40, 40, 235), "yellow": (0, 200, 255), "green": (80, 210, 60)}
        for i, st in enumerate(("red", "yellow", "green")):
            c, _ = self.proj(np.array([[d - 0.05, lat, 5.35 - i * 0.42]]))
            r = max(2, int(self.f * 0.15 / max(1.0, d + 11)))
            cv2.circle(img, tuple(np.int32(c[0])), r, colors[st] if st == state else (95, 95, 100), -1, cv2.LINE_AA)


# =====================================================================
# UI panels
# =====================================================================
UI_BG, UI_PANEL, UI_EDGE = (24, 18, 14), (38, 30, 24), (70, 58, 48)
UI_TEXT, UI_DIM = (235, 235, 235), (160, 160, 165)
SEV_COL = {"info": (230, 200, 90), "warn": (0, 190, 255), "crit": (70, 70, 240)}
LANE_TAGS = {0: "EGO", 1: "L1", -1: "R1", 2: "L2", -2: "R2", 3: "L3", -3: "R3"}


def panel(canvas, x, y, w, h, title=None):
    rrect(canvas, x, y, w, h, 10, UI_PANEL)
    rrect(canvas, x, y, w, h, 10, UI_EDGE, 1)
    if title:
        put(canvas, title, x + 14, y + 24, 0.55, UI_TEXT, 1, FONT_B)


def fit_into(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, ox, oy


def annotate_camera(frame, geo, lane, lc, tm, t, w, h):
    img, s, ox, oy = fit_into(frame, w, h)
    if lane["valid"]:
        dg = geo.img_to_ground(geo.fw / 2, geo.fh - 2)
        dstart = max(1.0, dg[0]) if dg else geo.near
        ds = np.linspace(dstart, min(70.0, geo.far * 2.3), 60)
        dm = lane["far"]
        L, R = pv_ext(lane["cl"], ds, dm), pv_ext(lane["cr"], ds, dm)

        def pts(lat):
            uv, ok = geo.ground_to_img(ds, lat)
            uv = uv[ok] * s + [ox, oy]
            return uv.astype(np.int32)
        pl, pr = pts(L), pts(R)
        if len(pl) > 2 and len(pr) > 2:
            ov = img.copy()
            fill = (255, 170, 60) if lc.state != "KEEP" else ((0, 200, 255) if lane.get("junction") else (90, 220, 90))
            cv2.fillPoly(ov, [np.vstack([pl, pr[::-1]])], fill, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)
        segs = [(lane["type_L"], lane["yellow_L"], L, True), (lane["type_R"], lane["yellow_R"], R, True)]
        for ln in lane.get("lines", []):
            if not ln["ego"]:
                segs.append((ln["type"], ln["yellow"], pv_ext(ln["c"], ds, dm), False))
        for typ, yel, lat, is_ego in segs:
            P = pts(lat)
            if len(P) < 2:
                continue
            col = (0, 210, 255) if yel else ((255, 255, 255) if is_ego else (230, 200, 120))
            th = 3 if is_ego else 2
            if typ == "solid":
                cv2.polylines(img, [P], False, col, th, cv2.LINE_AA)
            else:
                for i in range(0, len(P) - 1, 4):
                    cv2.polylines(img, [P[i:i + 3]], False, col, th, cv2.LINE_AA)
    for tr in tm.tracks.values():
        if tr.missed > 0 or tr.bbox is None:
            continue
        x1, y1, x2, y2 = [int(v * s + o) for v, o in zip(tr.bbox, (ox, oy, ox, oy))]
        col = CLASS_COLORS.get(tr.kind, (180, 180, 180))
        thick = 2
        if tr.id == tm.lead_id: col, thick = (255, 140, 40), 3
        if t < tr.flag_until or tr.intent: col, thick = (30, 140, 255), 3
        if tr.ttc is not None and tr.ttc < 2.5 and tr.lane == 0: col, thick = (60, 60, 240), 3
        cl = max(6, min(x2 - x1, y2 - y1) // 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * cl, py), col, thick, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * cl), col, thick, cv2.LINE_AA)
        lbl = f"{tr.cls} {tr.conf:.2f}"
        if tr.kind == "light":
            lbl += f" {tr.light.upper()}"
        elif tr.d is not None:
            lbl += f" | {tr.d:.1f}m | {LANE_TAGS.get(tr.lane, '-')}"
        tw = text_w(lbl, 0.4) + 8
        cv2.rectangle(img, (x1, max(0, y1 - 16)), (x1 + tw, max(16, y1)), col, -1)
        put(img, lbl, x1 + 4, max(12, y1 - 4), 0.4, (15, 15, 15))
    return img


def bev_panel(geo, lane, lane_det, tm, w, h):
    if lane_det.last_binary is None:
        return np.zeros((h, w, 3), np.uint8)
    bev = np.zeros((geo.bh, geo.bw, 3), np.uint8)
    bev[lane_det.last_binary > 0] = (235, 235, 235)
    bev[(lane_det.last_binary > 0) & (lane_det.last_yellow > 0)] = (0, 210, 255)
    if lane_det.last_reject is not None:
        bev[lane_det.last_reject > 0] = (60, 60, 150)          # rejected: crosswalk / arrows / stop line
    ds = np.linspace(geo.d0, geo.d1, 40)
    for ln in lane.get("lines", []):
        u, v = geo.metric_to_bev(ds, pv(ln["c"], ds))
        col = (90, 230, 90) if ln["ego"] else (230, 170, 60)
        cv2.polylines(bev, [np.int32(np.stack([u, v], 1))], False, col, 2, cv2.LINE_AA)
    for tr in tm.tracks.values():
        if tr.d is None or tr.kind == "light" or tr.missed > 3:
            continue
        u, v = geo.metric_to_bev(tr.d, tr.lat)
        col = (255, 140, 40) if tr.id == tm.lead_id else CLASS_COLORS.get(tr.kind, (200, 200, 200))
        cv2.rectangle(bev, (int(u - 9), int(v - 18)), (int(u + 9), int(v)), col, -1)
    u, v = geo.metric_to_bev(geo.d0 + 1.0, 0.0)
    cv2.rectangle(bev, (int(u - 10), int(v - 6)), (int(u + 10), int(v + 20)), (60, 160, 255), -1)
    out, *_ = fit_into(bev, w, h)
    return out


def compose(canvas, twin, cam, bev, lane, lc, tm, events, t, fidx, total, speed, proc_fps, video_name):
    canvas[:] = UI_BG
    put(canvas, "ADAS DIGITAL TWIN", 20, 38, 0.95, UI_TEXT, 2, FONT_B)
    put(canvas, "Video -> Perception -> Lanes -> Tracking -> Scenario -> 3D Twin", 330, 36, 0.55, UI_DIM)
    put(canvas, f"{video_name}  |  t {t:6.2f} s  |  frame {fidx}/{total}  |  {proc_fps:4.1f} fps",
        1030, 36, 0.5, UI_DIM)
    blink = int(t * 3) % 2 == 0
    junction = lane.get("junction")
    chip_col = (40, 150, 255) if lc.state != "KEEP" else ((0, 200, 255) if junction else (90, 200, 90))
    chip_txt = ("JUNCTION" if junction else "LANE KEEP") if lc.state == "KEEP" else lc.state.replace("_", " ")
    tw = text_w(chip_txt, 0.6, 2) + 30
    rrect(canvas, OUT_W - tw - 20, 14, tw, 34, 17, chip_col)
    put(canvas, chip_txt, OUT_W - tw - 5, 38, 0.6, (20, 20, 20), 2)

    tx, ty = 12, 64
    canvas[ty:ty + twin.shape[0], tx:tx + twin.shape[1]] = twin
    rrect(canvas, tx, ty, twin.shape[1], twin.shape[0], 6, UI_EDGE, 1)
    tc = PALETTES[THEME]["text"]
    if speed is not None:
        put(canvas, f"{speed * 3.6:.0f}", tx + 28, ty + 78, 2.0, tc, 3, FONT_B)
        put(canvas, "km/h EST" if EGO_SPEED_KPH is None else "km/h SET", tx + 32, ty + 104, 0.55, tc)
    else:
        put(canvas, "--", tx + 28, ty + 78, 2.0, tc, 3, FONT_B)
        put(canvas, "speed n/a", tx + 32, ty + 104, 0.55, tc)
    if lc.state != "KEEP":
        bw, bh = 540, 54
        bx, by = tx + twin.shape[1] // 2 - bw // 2, ty + 20
        rrect(canvas, bx, by, bw, bh, 27, (40, 150, 255) if blink else (30, 110, 200))
        d = lc.dir
        ax = bx + 40 if d > 0 else bx + bw - 40
        arrow = np.array([[ax - 18 * d, by + 27], [ax + 6 * d, by + 9], [ax + 6 * d, by + 19], [ax + 22 * d, by + 19],
                          [ax + 22 * d, by + 35], [ax + 6 * d, by + 35], [ax + 6 * d, by + 45]], np.int32)
        if blink:
            cv2.fillPoly(canvas, [arrow], (255, 255, 255), cv2.LINE_AA)
        txt = f"LANE CHANGE {'LEFT' if d > 0 else 'RIGHT'}" + ("  - CROSSED" if lc.crossed else "")
        tx0 = bx + 80 if d > 0 else bx + bw - 80 - text_w(txt, 0.7, 2)
        put(canvas, txt, tx0, by + 36, 0.7, (255, 255, 255), 2, FONT_B)
    lx, ly = tx + 20, ty + twin.shape[0] - 30
    pal = PALETTES[THEME]
    legend_items = [
        ("Lead", pal["lead"]),
        ("Cut-in / lane change", pal["cut"]),
        ("Critical", pal["crit"]),
        ("Other", pal["other"]),
    ]
    if SHOW_PATH_PLANNING:
        legend_items.append(("Ego path", pal["path"]))
    for name, col in legend_items:
        rrect(canvas, lx, ly - 12, 16, 16, 4, col)
        put(canvas, name, lx + 22, ly + 1, 0.45, tc)
        lx += text_w(name, 0.45) + 50

    cx, cy, cw, ch = 1124, 64, 784, 441
    canvas[cy:cy + ch, cx:cx + cw] = cam
    rrect(canvas, cx, cy, cw, ch, 6, UI_EDGE, 1)
    blend_rect(canvas, cx, cy, 250, 30, (0, 0, 0), 0.55)
    put(canvas, "FRONT CAMERA | PERCEPTION", cx + 10, cy + 21, 0.5, UI_TEXT)

    px, py, pw, ph = 1124, 517, 300, 330
    panel(canvas, px, py, pw, ph, "Lanes (BEV)")
    canvas[py + 36:py + 36 + 210, px + 10:px + 10 + 280] = bev
    yy = py + 270
    if lane["valid"]:
        def lt(k):
            return f"{lane['type_' + k]} {'yellow' if lane['yellow_' + k] else 'white'}".upper() if lane["type_" + k] != "none" else "NOT SEEN"
        put(canvas, f"L: {lt('L')}", px + 14, yy, 0.45, UI_TEXT)
        put(canvas, f"R: {lt('R')}", px + 14, yy + 20, 0.45, UI_TEXT)
        put(canvas, f"Lanes {lane.get('n_lanes', 0)} | ego lane {lane.get('ego_lane_no', 0)} | W {lane['w']:.2f} m"
            + ("  JUNCTION" if junction else ""), px + 14, yy + 40, 0.42, (0, 200, 255) if junction else UI_DIM)
        cv2.rectangle(canvas, (px + 14, yy + 50), (px + 214, yy + 56), (70, 60, 50), -1)
        cv2.rectangle(canvas, (px + 14, yy + 50), (px + 14 + int(200 * lane["conf"]), yy + 56), (90, 220, 90), -1)
        put(canvas, f"{lane['conf'] * 100:.0f}%", px + 222, yy + 57, 0.4, UI_DIM)
    else:
        put(canvas, "Lane not detected", px + 14, yy, 0.5, (60, 60, 240))

    ox, oy, ow, oh = 1436, 517, 472, 330
    panel(canvas, ox, oy, ow, oh, "Objects (nearest)")
    cols = [("ID", 0), ("Class", 46), ("Lane", 160), ("Dist", 222), ("RelV", 292), ("TTC", 370)]
    for name, dx in cols:
        put(canvas, name, ox + 14 + dx, oy + 52, 0.45, UI_DIM)
    rows = sorted([tr for tr in tm.tracks.values() if tr.missed == 0 and tr.d is not None and tr.kind != "light"],
                  key=lambda r: r.d)[:10]
    for i, tr in enumerate(rows):
        y = oy + 78 + i * 24
        col = (255, 160, 60) if tr.id == tm.lead_id else UI_TEXT
        vals = [f"{tr.id}", tr.cls[:12], LANE_TAGS.get(tr.lane, "-"), f"{tr.d:5.1f} m", f"{tr.vd:+5.1f}",
                "-" if tr.ttc is None else f"{tr.ttc:4.1f} s"]
        for (name, dx), v in zip(cols, vals):
            put(canvas, v, ox + 14 + dx, y, 0.45, col)
        cv2.circle(canvas, (ox + ow - 18, y - 5), 5, CLASS_COLORS.get(tr.kind, (180, 180, 180)), -1, cv2.LINE_AA)
    put(canvas, f"Tracks {len(rows)} | Lane changes L{lc.count['LEFT']} / R{lc.count['RIGHT']}",
        ox + 14, oy + oh - 12, 0.45, UI_DIM)

    ex, ey, ew, eh = 1124, 859, 784, 209
    panel(canvas, ex, ey, ew, eh, "Events")
    for i, e in enumerate(events.items[-7:][::-1]):
        y = ey + 52 + i * 22
        cv2.circle(canvas, (ex + 20, y - 5), 5, SEV_COL.get(e["severity"], UI_TEXT), -1, cv2.LINE_AA)
        put(canvas, f"{e['time_s']:7.2f} s", ex + 34, y, 0.45, UI_DIM)
        put(canvas, e["message"], ex + 120, y, 0.45, UI_TEXT)
    put(canvas, "Monocular reconstruction - distances depend on calibration. Not a safety validation.",
        20, OUT_H - 2, 0.38, (120, 120, 125))
    return canvas


# =====================================================================
# Pipeline
# =====================================================================
def run_pipeline(video: Path, out: Path, trap=None, near=NEAR_M, far=FAR_M, detector=None, H=None):
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if MAX_SECONDS:
        total = min(total, int(MAX_SECONDS * fps))
    und = make_undistorter()
    geo = GroundGeometry(fw, fh, trap or DEFAULT_TRAPEZOID, near, far, LANE_WIDTH_M, H=H)
    detector = detector or YoloTracker()
    lane_det = LaneDetector(geo, LANE_WIDTH_M, fps=fps / FRAME_STRIDE)
    speed_est = EgoSpeedEstimator(geo)
    lc = LaneChangeMonitor(fps / FRAME_STRIDE)
    tm = TrackManager(fps)
    events = EventLog()
    renderer = TwinRenderer(1100, 1004, SUPERSAMPLE, THEME)
    renderer._src_w = fw
    writer = cv2.VideoWriter(str(out / "adas_digital_twin_hd.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / FRAME_STRIDE, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError("Cannot create output video")
    ft = (out / "tracks.csv").open("w", newline="", encoding="utf-8")
    fl = (out / "lanes.csv").open("w", newline="", encoding="utf-8")
    tw_ = csv.writer(ft); lw_ = csv.writer(fl)
    tw_.writerow(["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "d_m", "lat_m",
                  "lane", "rel_vd_mps", "vlat_mps", "ttc_s", "is_lead", "light_state"])
    lw_.writerow(["frame", "time_s", "valid", "left_a", "left_b", "left_c", "right_a", "right_b", "right_c",
                  "width_m", "conf", "left_type", "right_type", "left_yellow", "right_yellow",
                  "n_lanes", "ego_lane_no", "junction", "ego_offset_m", "ego_lat_vel_mps", "lc_state",
                  "ego_speed_kph_est"])
    canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    dt = FRAME_STRIDE / fps
    fidx, processed, t0 = 0, 0, time.time()
    lane_conf_sum, class_frames = 0.0, Counter()
    paused = False
    try:
        while fidx < total or total <= 0:
            ok, raw = cap.read()
            if not ok:
                break
            if fidx % FRAME_STRIDE:
                fidx += 1
                continue
            t = fidx / fps
            frame = und(raw)
            dets = detector.detect(frame)
            block = np.zeros((fh, fw), np.uint8)
            for dd in dets:
                x1, y1, x2, y2 = [int(v) for v in dd["bbox"]]
                cv2.rectangle(block, (x1 - 4, y1), (x2 + 4, y2 + 4), 255, -1)
            bev = cv2.warpPerspective(frame, geo.M, (geo.bw, geo.bh), flags=cv2.INTER_LINEAR)
            bblock = cv2.warpPerspective(block, geo.M, (geo.bw, geo.bh), flags=cv2.INTER_NEAREST)
            lane = lane_det.update(bev, bblock)
            speed = speed_est.update(bev, bblock, dt)
            lc.update(t, lane, events, fidx)
            tm.update(dets, frame, geo, lane, t, fidx, dt, events)

            twin = renderer.render(lane, lc, tm.tracks, tm.lead_id, speed, t, dt)
            cam = annotate_camera(frame, geo, lane, lc, tm, t, 784, 441)
            bv = bev_panel(geo, lane, lane_det, tm, 280, 210)
            proc_fps = (processed + 1) / max(1e-3, time.time() - t0)
            compose(canvas, twin, cam, bv, lane, lc, tm, events, t, fidx, total, speed, proc_fps, video.name)
            writer.write(canvas)

            lane_conf_sum += lane["conf"]
            for tr in tm.tracks.values():
                if tr.missed:
                    continue
                class_frames[tr.cls] += 1
                b = tr.bbox or (0, 0, 0, 0)
                tw_.writerow([fidx, f"{t:.3f}", tr.id, tr.cls, f"{tr.conf:.3f}", *[f"{v:.1f}" for v in b],
                              "" if tr.d is None else f"{tr.d:.2f}", "" if tr.lat is None else f"{tr.lat:.2f}",
                              "" if tr.lane is None else tr.lane, f"{tr.vd:.2f}", f"{tr.vlat:.2f}",
                              "" if tr.ttc is None else f"{tr.ttc:.2f}", int(tr.id == tm.lead_id),
                              tr.light if tr.kind == "light" else ""])
            cl_ = [f"{v:.6f}" for v in lane["cl"]] if lane["valid"] else [""] * 3
            cr_ = [f"{v:.6f}" for v in lane["cr"]] if lane["valid"] else [""] * 3
            lw_.writerow([fidx, f"{t:.3f}", int(lane["valid"]), *cl_, *cr_, f"{lane['w']:.2f}", f"{lane['conf']:.2f}",
                          lane["type_L"], lane["type_R"], int(lane["yellow_L"]), int(lane["yellow_R"]),
                          lane["n_lanes"], lane["ego_lane_no"], int(lane["junction"]),
                          f"{lc.off:.3f}", f"{lc.vel:.3f}", lc.state, "" if speed is None else f"{speed * 3.6:.1f}"])
            processed += 1
            fidx += 1
            if processed % 50 == 0:
                LOG.info("Frame %d/%d  (%.1f fps)", fidx, total, proc_fps)
            if SHOW_PREVIEW:
                try:
                    cv2.imshow("ADAS Digital Twin (q=quit, p=pause)", cv2.resize(canvas, (1280, 720)))
                    k = cv2.waitKey(0 if paused else 1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("p"):
                        paused = not paused
                except cv2.error:
                    pass
    finally:
        cap.release(); writer.release(); ft.close(); fl.close()
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    (out / "events.json").write_text(json.dumps(events.items, indent=2), encoding="utf-8")
    with (out / "events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time_s", "frame", "type", "severity", "message"])
        w.writeheader(); w.writerows(events.items)
    summary = {
        "video": str(video), "fps": fps, "processed_frames": processed, "frame_size": [fw, fh],
        "lane_calibration": {"mode": "auto" if H is not None else ("manual" if trap else "default"),
                             "near_m": near, "far_m": far},
        "mean_lane_confidence": round(lane_conf_sum / max(1, processed), 3),
        "ego_lane_changes": lc.count,
        "actor_lane_events": len([e for e in events.items if e["type"] == "actor_lane_change"]),
        "class_track_frames": dict(class_frames),
        "outputs": ["adas_digital_twin_hd.mp4", "tracks.csv", "lanes.csv", "events.csv", "events.json"],
        "note": "Monocular reconstruction; metric values depend on calibration. Not a safety validation.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Done. Results in %s", out)
    return summary


# =====================================================================
# Manual lane calibration (click 4 points) and popups
# =====================================================================
def order_points(pts):
    p = sorted(pts, key=lambda q: q[1])
    top, bot = sorted(p[:2], key=lambda q: q[0]), sorted(p[2:], key=lambda q: q[0])
    return [bot[0], bot[1], top[1], top[0]]          # BL, BR, TR, TL


def calibrate_interactive(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    und = make_undistorter()
    pos = min(int(2 * fps), total - 1)

    def read(p):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, fr = cap.read()
        return und(fr) if ok else None
    frame = read(pos)
    if frame is None:
        cap.release(); return None
    fh, fw = frame.shape[:2]
    sc = min(1.0, 1280 / fw, 760 / fh)
    pts = []
    win = "Lane calibration"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / sc, y / sc))
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        disp = cv2.resize(frame, None, fx=sc, fy=sc)
        blend_rect(disp, 0, 0, disp.shape[1], 70, (0, 0, 0), 0.6)
        put(disp, "Straight road frame: click 4 points on EGO-LANE markings - 2 near (bottom), 2 far (top)",
            10, 25, 0.55, (255, 255, 255))
        put(disp, "n/b = next/back 1 s | r = reset | ENTER = confirm | ESC = cancel", 10, 52, 0.5, (0, 220, 255))
        for p in pts:
            cv2.circle(disp, (int(p[0] * sc), int(p[1] * sc)), 6, (0, 255, 0), -1, cv2.LINE_AA)
        if len(pts) == 4:
            cv2.polylines(disp, [np.int32(np.array(order_points(pts)) * sc)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("n"), ord("b")):
            pos = min(total - 1, pos + int(fps)) if k == ord("n") else max(0, pos - int(fps))
            nf = read(pos)
            frame = nf if nf is not None else frame
        elif k == ord("r"):
            pts.clear()
        elif k in (13, 32) and len(pts) == 4:
            result = [(x / fw, y / fh) for x, y in order_points(pts)]
            break
        elif k == 27:
            break
    cv2.destroyWindow(win)
    cap.release()
    return result


def ask_inputs():
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    try:
        v = filedialog.askopenfilename(parent=root, title="Step 1: Select road video",
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All", "*.*")])
        if not v:
            return None
        video = Path(v)
        o = filedialog.askdirectory(parent=root, title="Step 2: Select output folder", initialdir=str(video.parent))
        if not o:
            return None
        out = Path(o) / ("digital_twin_hd_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        trap, near, far, H = None, NEAR_M, FAR_M, None
        saved = video.with_name(video.stem + "_lane_calib.json")
        if saved.exists() and messagebox.askyesno("Lane calibration", f"Use saved calibration?\n{saved.name}", parent=root):
            c = json.loads(saved.read_text(encoding="utf-8"))
            near, far = c["near_m"], c["far_m"]
            if c.get("H") is not None:
                H = np.asarray(c["H"], np.float64)
            else:
                trap = [tuple(x) for x in c["trapezoid"]]
            return video, out, trap, near, far, H
        choice = True
        if ASK_LANE_CALIBRATION:
            choice = messagebox.askyesnocancel(
                "Lane calibration",
                "YES  = AUTOMATIC calibration (vanishing point + lane width) - recommended\n"
                "NO   = MANUAL: click 4 points on the ego-lane markings\n"
                "CANCEL = default camera assumptions", parent=root)
        if choice is True:
            H, near, far, _ = auto_calibrate(video, make_undistorter())
            data = {"H": H.tolist(), "near_m": near, "far_m": far}
        elif choice is False:
            trap = calibrate_interactive(video)
            if not trap:
                return video, out, None, NEAR_M, FAR_M, None
            near = simpledialog.askfloat("Near distance", "Ground distance (m) from camera to the 2 BOTTOM points:",
                                         initialvalue=NEAR_M, minvalue=1.0, maxvalue=40.0, parent=root) or NEAR_M
            far = simpledialog.askfloat("Far distance", "Ground distance (m) from camera to the 2 TOP points:\n"
                                        "(tip: one dash + gap is typically ~9-12 m)",
                                        initialvalue=FAR_M, minvalue=near + 3, maxvalue=120.0, parent=root) or FAR_M
            data = {"trapezoid": trap, "near_m": near, "far_m": far}
        else:
            return video, out, None, NEAR_M, FAR_M, None
        try:
            saved.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass
        return video, out, trap, near, far, H
    finally:
        root.destroy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sel = ask_inputs()
    if sel is None:
        print("Selection cancelled. Nothing processed.")
        return
    video, out, trap, near, far, H = sel
    print("Processing:", video)
    summary = run_pipeline(video, out, trap, near, far, H=H)
    print(json.dumps(summary, indent=2))
    print("Output video:", out / "adas_digital_twin_hd.mp4")


if __name__ == "__main__":
    main()

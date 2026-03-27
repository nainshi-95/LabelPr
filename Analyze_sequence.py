#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML not installed. Install: pip install pyyaml") from e


# ============================================================
# YAML + seq cfg parsing
# ============================================================
_CFG_LINE_RE = __import__("re").compile(r"^\s*([^:#]+?)\s*:\s*(.*?)\s*$")


def load_yaml_dict(yaml_path: Path) -> Dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError("YAML root must be dict.")
    return d


def parse_seq_cfg(seq_cfg_path: Path) -> Dict[str, str]:
    if not seq_cfg_path.is_file():
        raise FileNotFoundError(seq_cfg_path)

    lines = seq_cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return {}

    out: Dict[str, str] = {}
    for raw in lines[1:]:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].rstrip()

        m = _CFG_LINE_RE.match(s)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        if k:
            out[k] = v
    return out


def _pick(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def collect_seq_items_from_yaml(
    yaml_path: Path,
    only_seq: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    y = load_yaml_dict(yaml_path)

    if "seq" not in y or not isinstance(y["seq"], dict):
        raise KeyError("YAML missing seq dict")

    defaults = y.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be dict if provided")

    seq_dict: Dict[str, Dict] = y["seq"]
    items: List[Dict[str, Any]] = []

    for seq_name, info in seq_dict.items():
        if not isinstance(info, dict):
            continue
        if only_seq is not None and seq_name not in only_seq:
            continue

        seq_cls = str(info.get("seq_cls", "NA"))
        seq_cfg = _pick(info, ["seq_cfg"], _pick(defaults, ["seq_cfg"], None))

        width = _pick(info, ["width", "w", "source_width"], _pick(defaults, ["width"], None))
        height = _pick(info, ["height", "h", "source_height"], _pick(defaults, ["height"], None))
        frames = _pick(info, ["frames", "num_frames", "FrameToBeEncoded"], _pick(defaults, ["frames"], None))
        bit_depth = _pick(info, ["bit_depth", "bitdepth", "input_bit_depth"], _pick(defaults, ["bit_depth"], None))
        fps = _pick(info, ["fps", "frame_rate", "FrameRate"], _pick(defaults, ["fps"], None))
        yuv_path = _pick(info, ["path", "yuv_path", "yuv"], _pick(defaults, ["path"], None))

        if seq_cfg:
            cfg_path = Path(str(seq_cfg))
            if cfg_path.is_file():
                cfg = parse_seq_cfg(cfg_path)
                if width is None:
                    try:
                        width = int(cfg.get("SourceWidth", cfg.get("InputFileWidth", "")))
                    except Exception:
                        pass
                if height is None:
                    try:
                        height = int(cfg.get("SourceHeight", cfg.get("InputFileHeight", "")))
                    except Exception:
                        pass
                if frames is None:
                    try:
                        frames = int(cfg.get("FramesToBeEncoded", cfg.get("FrameToBeEncoded", "")))
                    except Exception:
                        pass
                if fps is None:
                    try:
                        fps = float(cfg.get("FrameRate", "30"))
                    except Exception:
                        pass
                if bit_depth is None:
                    try:
                        bit_depth = int(cfg.get("InputBitDepth", cfg.get("BitDepth", "10")))
                    except Exception:
                        pass
                if yuv_path is None:
                    yuv_in_cfg = cfg.get("InputFile", "").strip()
                    if yuv_in_cfg:
                        yuv_path = yuv_in_cfg

        if yuv_path is None:
            raise ValueError(f"Missing yuv path for seq={seq_name}")
        if width is None or height is None:
            raise ValueError(f"Missing width/height for seq={seq_name}")
        if frames is None:
            raise ValueError(f"Missing frames for seq={seq_name}")
        if fps is None:
            fps = 30.0
        if bit_depth is None:
            bit_depth = 10

        items.append({
            "name": str(seq_name),
            "seq_cls": str(seq_cls),
            "yuv_path": str(yuv_path),
            "width": int(width),
            "height": int(height),
            "frames": int(frames),
            "frame_rate": float(fps),
            "bit_depth": int(bit_depth),
        })

    if not items:
        raise RuntimeError("No valid sequences found in YAML after filters.")

    return items


# ============================================================
# YUV420 raw IO
# ============================================================
def read_yuv420p_raw_y_only(
    path: Path,
    width: int,
    height: int,
    num_frames: int,
    bit_depth: int,
) -> np.ndarray:
    """
    Returns:
      Y: [T, H, W]
    """
    if bit_depth <= 8:
        sample_dtype = np.uint8
        bytes_per_sample = 1
    else:
        sample_dtype = np.dtype("<u2")
        bytes_per_sample = 2

    w2, h2 = width // 2, height // 2
    y_n = width * height
    uv_n = w2 * h2
    frame_bytes = (y_n + uv_n + uv_n) * bytes_per_sample

    Y = np.empty((num_frames, height, width), dtype=sample_dtype)

    with open(path, "rb") as f:
        for t in range(num_frames):
            yb = f.read(y_n * bytes_per_sample)
            if len(yb) != y_n * bytes_per_sample:
                raise IOError(f"EOF while reading Y plane {path} at frame {t}")
            Y[t] = np.frombuffer(yb, dtype=sample_dtype).reshape(height, width)

            # skip U/V
            skipped = f.seek(2 * uv_n * bytes_per_sample, 1)
            _ = skipped

    return Y


def to_float01(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    maxv = float((1 << bit_depth) - 1)
    out = arr.astype(np.float32) / maxv
    return np.clip(out, 0.0, 1.0)


# ============================================================
# Utility
# ============================================================
def determine_analysis_unit(frame_idx: int) -> Optional[int]:
    if frame_idx % 16 == 0:
        return 16
    if frame_idx % 8 == 0:
        return 8
    return None


def make_tag(seq_name: str, frame_idx: int, analysis_unit: int) -> str:
    return f"{seq_name}_t{frame_idx:04d}_u{analysis_unit}"


def reflect_index_1d(i: int, n: int) -> int:
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


def get_temporal_neighbors(center: int, count_each_side: int, total_frames: int) -> List[int]:
    idxs = []
    for k in range(-count_each_side, count_each_side + 1):
        if k == 0:
            continue
        idxs.append(reflect_index_1d(center + k, total_frames))
    return idxs


def block_view_2d(img: np.ndarray, block_size: int) -> np.ndarray:
    """
    img: [H, W]
    return: [nBH, nBW, block_size, block_size]
    Crops tail if needed.
    """
    H, W = img.shape
    Hc = (H // block_size) * block_size
    Wc = (W // block_size) * block_size
    x = img[:Hc, :Wc]
    nBH = Hc // block_size
    nBW = Wc // block_size
    x = x.reshape(nBH, block_size, nBW, block_size).transpose(0, 2, 1, 3)
    return x


def satd_4x4(block: np.ndarray) -> float:
    """
    Simple 4x4 Hadamard SATD for one block.
    block: [4,4], float32 or float64
    """
    x = block.astype(np.float32)

    m = np.empty((4, 4), dtype=np.float32)
    m[0] = x[0] + x[3]
    m[1] = x[1] + x[2]
    m[2] = x[1] - x[2]
    m[3] = x[0] - x[3]

    t = np.empty((4, 4), dtype=np.float32)
    t[:, 0] = m[:, 0] + m[:, 1]
    t[:, 1] = m[:, 3] + m[:, 2]
    t[:, 2] = m[:, 0] - m[:, 1]
    t[:, 3] = m[:, 3] - m[:, 2]

    m2 = np.empty((4, 4), dtype=np.float32)
    m2[0] = t[:, 0] + t[:, 3]
    m2[1] = t[:, 1] + t[:, 2]
    m2[2] = t[:, 1] - t[:, 2]
    m2[3] = t[:, 0] - t[:, 3]

    t2 = np.empty((4, 4), dtype=np.float32)
    t2[:, 0] = m2[:, 0] + m2[:, 1]
    t2[:, 1] = m2[:, 3] + m2[:, 2]
    t2[:, 2] = m2[:, 0] - m2[:, 1]
    t2[:, 3] = m2[:, 3] - m2[:, 2]

    return float(np.sum(np.abs(t2)) / 2.0)


def satd_image_4x4(diff: np.ndarray) -> float:
    """
    Mean SATD over all non-overlapping 4x4 blocks.
    """
    H, W = diff.shape
    Hc = (H // 4) * 4
    Wc = (W // 4) * 4
    if Hc == 0 or Wc == 0:
        return float(np.mean(np.abs(diff)))

    d = diff[:Hc, :Wc]
    total = 0.0
    count = 0
    for y in range(0, Hc, 4):
        for x in range(0, Wc, 4):
            total += satd_4x4(d[y:y+4, x:x+4])
            count += 1
    return total / max(count, 1)


def integer_pel_motion_compensate(
    cur: np.ndarray,
    ref: np.ndarray,
    block_size: int = 16,
    search_range: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very simple integer-pel block matching.
    Returns:
      pred: [Hc, Wc]
      sad_map: [nBH, nBW] best SAD per block
    """
    H, W = cur.shape
    Hc = (H // block_size) * block_size
    Wc = (W // block_size) * block_size
    cur_c = cur[:Hc, :Wc]
    ref_c = ref[:Hc, :Wc]

    pred = np.empty_like(cur_c)
    nBH = Hc // block_size
    nBW = Wc // block_size
    sad_map = np.empty((nBH, nBW), dtype=np.float32)

    for by in range(nBH):
        y = by * block_size
        for bx in range(nBW):
            x = bx * block_size
            blk = cur_c[y:y+block_size, x:x+block_size]

            best_sad = None
            best_patch = None

            for dy in range(-search_range, search_range + 1):
                yy = min(max(y + dy, 0), Hc - block_size)
                for dx in range(-search_range, search_range + 1):
                    xx = min(max(x + dx, 0), Wc - block_size)
                    ref_blk = ref_c[yy:yy+block_size, xx:xx+block_size]
                    sad = float(np.sum(np.abs(blk - ref_blk)))
                    if best_sad is None or sad < best_sad:
                        best_sad = sad
                        best_patch = ref_blk

            pred[y:y+block_size, x:x+block_size] = best_patch
            sad_map[by, bx] = best_sad

    return pred, sad_map


# ============================================================
# Analyzer interface
# ============================================================
class BaseAnalyzer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def required_radius(self, analysis_unit: int) -> int:
        """
        How many frames on each side may be needed.
        """
        pass

    @abstractmethod
    def compute(
        self,
        Y: np.ndarray,
        center_idx: int,
        analysis_unit: int,
        seq_item: Dict[str, Any],
    ) -> Dict[str, float]:
        pass


class BlockVarianceAnalyzer(BaseAnalyzer):
    def __init__(self, block_sizes: List[int]):
        self.block_sizes = block_sizes

    @property
    def name(self) -> str:
        return "block_variance"

    def required_radius(self, analysis_unit: int) -> int:
        return 0

    def compute(
        self,
        Y: np.ndarray,
        center_idx: int,
        analysis_unit: int,
        seq_item: Dict[str, Any],
    ) -> Dict[str, float]:
        img = Y[center_idx]
        out: Dict[str, float] = {}

        for bs in self.block_sizes:
            blocks = block_view_2d(img, bs)
            if blocks.size == 0:
                out[f"var_bs{bs}_mean"] = np.nan
                out[f"var_bs{bs}_std"] = np.nan
                out[f"var_bs{bs}_p90"] = np.nan
                continue

            vars_ = blocks.var(axis=(2, 3))
            out[f"var_bs{bs}_mean"] = float(vars_.mean())
            out[f"var_bs{bs}_std"] = float(vars_.std())
            out[f"var_bs{bs}_p90"] = float(np.percentile(vars_, 90))
            out[f"var_bs{bs}_p10"] = float(np.percentile(vars_, 10))

        return out


class TemporalDiffAnalyzer(BaseAnalyzer):
    def __init__(self, neighbors_each_side: int = 4):
        self.neighbors_each_side = neighbors_each_side

    @property
    def name(self) -> str:
        return "temporal_diff"

    def required_radius(self, analysis_unit: int) -> int:
        return self.neighbors_each_side

    def compute(
        self,
        Y: np.ndarray,
        center_idx: int,
        analysis_unit: int,
        seq_item: Dict[str, Any],
    ) -> Dict[str, float]:
        T = Y.shape[0]
        cur = Y[center_idx]
        idxs = get_temporal_neighbors(center_idx, self.neighbors_each_side, T)

        vals = []
        for idx in idxs:
            diff = np.abs(cur - Y[idx])
            vals.append(float(diff.mean()))

        arr = np.array(vals, dtype=np.float32)
        return {
            "tdiff_mean": float(arr.mean()),
            "tdiff_std": float(arr.std()),
            "tdiff_min": float(arr.min()),
            "tdiff_max": float(arr.max()),
        }


class McSatdAnalyzer(BaseAnalyzer):
    """
    Simple example analyzer:
      current frame vs past/future 4 neighbors
      integer-pel motion compensation
      SATD of residual
      mean/std aggregation
    """
    def __init__(
        self,
        neighbors_each_side: int = 4,
        mc_block_size: int = 16,
        search_range: int = 8,
    ):
        self.neighbors_each_side = neighbors_each_side
        self.mc_block_size = mc_block_size
        self.search_range = search_range

    @property
    def name(self) -> str:
        return "mc_satd"

    def required_radius(self, analysis_unit: int) -> int:
        return self.neighbors_each_side

    def compute(
        self,
        Y: np.ndarray,
        center_idx: int,
        analysis_unit: int,
        seq_item: Dict[str, Any],
    ) -> Dict[str, float]:
        T = Y.shape[0]
        cur = Y[center_idx]
        idxs = get_temporal_neighbors(center_idx, self.neighbors_each_side, T)

        satd_vals = []
        sad_vals = []

        for idx in idxs:
            ref = Y[idx]
            pred, sad_map = integer_pel_motion_compensate(
                cur=cur,
                ref=ref,
                block_size=self.mc_block_size,
                search_range=self.search_range,
            )
            Hc, Wc = pred.shape
            diff = cur[:Hc, :Wc] - pred
            satd_vals.append(satd_image_4x4(diff))
            sad_vals.append(float(np.mean(sad_map)))

        satd_arr = np.array(satd_vals, dtype=np.float32)
        sad_arr = np.array(sad_vals, dtype=np.float32)

        return {
            "mc_satd_mean": float(satd_arr.mean()),
            "mc_satd_std": float(satd_arr.std()),
            "mc_satd_min": float(satd_arr.min()),
            "mc_satd_max": float(satd_arr.max()),
            "mc_sad_mean": float(sad_arr.mean()),
            "mc_sad_std": float(sad_arr.std()),
        }


# ============================================================
# Analyzer factory
# ============================================================
def build_analyzers_from_args(args) -> List[BaseAnalyzer]:
    analyzers: List[BaseAnalyzer] = []

    requested = [x.strip() for x in args.analyzers.split(",") if x.strip()]
    requested_set = set(requested)

    if "block_variance" in requested_set:
        analyzers.append(
            BlockVarianceAnalyzer(block_sizes=args.var_block_sizes)
        )

    if "temporal_diff" in requested_set:
        analyzers.append(
            TemporalDiffAnalyzer(neighbors_each_side=args.temporal_neighbors_each_side)
        )

    if "mc_satd" in requested_set:
        analyzers.append(
            McSatdAnalyzer(
                neighbors_each_side=args.mc_neighbors_each_side,
                mc_block_size=args.mc_block_size,
                search_range=args.mc_search_range,
            )
        )

    if not analyzers:
        raise ValueError("No analyzers selected.")

    return analyzers


# ============================================================
# Main analysis
# ============================================================
def analyze_sequence(
    seq_item: Dict[str, Any],
    analyzers: List[BaseAnalyzer],
    frame_limit: int = 0,
    start_frame: int = 0,
    end_frame: int = -1,
) -> List[Dict[str, Any]]:
    seq_name = seq_item["name"]
    seq_cls = seq_item["seq_cls"]
    yuv_path = Path(seq_item["yuv_path"])
    width = int(seq_item["width"])
    height = int(seq_item["height"])
    total_frames = int(seq_item["frames"])
    bit_depth = int(seq_item["bit_depth"])

    used_frames = total_frames if frame_limit <= 0 else min(frame_limit, total_frames)
    if end_frame >= 0:
        used_frames = min(used_frames, end_frame + 1)

    if start_frame < 0 or start_frame >= used_frames:
        raise ValueError(f"Invalid start_frame={start_frame} for seq={seq_name}")

    print(f"[INFO] Loading Y plane: {seq_name} ({used_frames} frames)")
    Y_raw = read_yuv420p_raw_y_only(
        path=yuv_path,
        width=width,
        height=height,
        num_frames=used_frames,
        bit_depth=bit_depth,
    )
    Y = to_float01(Y_raw, bit_depth)

    rows: List[Dict[str, Any]] = []

    last_frame = used_frames - 1
    for t in range(start_frame, last_frame + 1):
        unit = determine_analysis_unit(t)
        if unit is None:
            continue

        row: Dict[str, Any] = {
            "seq_name": seq_name,
            "seq_cls": seq_cls,
            "frame_idx": t,
            "analysis_unit": unit,
            "tag": make_tag(seq_name, t, unit),
            "range_start": max(0, t - unit // 2),
            "range_end": min(last_frame, t + unit // 2 - 1),
        }

        for analyzer in analyzers:
            vals = analyzer.compute(
                Y=Y,
                center_idx=t,
                analysis_unit=unit,
                seq_item=seq_item,
            )
            for k, v in vals.items():
                if k in row:
                    raise KeyError(f"Duplicate feature key detected: {k}")
                row[k] = v

        rows.append(row)

    return rows


# ============================================================
# CLI
# ============================================================
def parse_int_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Analyze YUV sequences at frames divisible by 8/16 and save all results to CSV"
    )

    parser.add_argument("--yaml", type=str, required=True, help="dataset yaml")
    parser.add_argument("--only_seq", type=str, default="", help="comma-separated seq names")
    parser.add_argument("--out_csv", type=str, required=True, help="output CSV path")

    parser.add_argument(
        "--analyzers",
        type=str,
        default="block_variance,temporal_diff",
        help="comma-separated analyzers: block_variance, temporal_diff, mc_satd",
    )

    parser.add_argument("--frame_limit", type=int, default=0, help="0 means all frames")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)

    # block variance
    parser.add_argument(
        "--var_block_sizes",
        type=str,
        default="8,16,32",
        help='e.g. "8,16,32"',
    )

    # temporal diff
    parser.add_argument("--temporal_neighbors_each_side", type=int, default=4)

    # mc satd
    parser.add_argument("--mc_neighbors_each_side", type=int, default=4)
    parser.add_argument("--mc_block_size", type=int, default=16)
    parser.add_argument("--mc_search_range", type=int, default=8)

    args = parser.parse_args()

    only_seq = {s.strip() for s in args.only_seq.split(",") if s.strip()}
    args.var_block_sizes = parse_int_list(args.var_block_sizes)

    seq_items = collect_seq_items_from_yaml(
        Path(args.yaml),
        only_seq=only_seq if only_seq else None,
    )
    analyzers = build_analyzers_from_args(args)

    print("[INFO] Selected analyzers:", [a.name for a in analyzers])

    all_rows: List[Dict[str, Any]] = []
    for seq_item in seq_items:
        rows = analyze_sequence(
            seq_item=seq_item,
            analyzers=analyzers,
            frame_limit=args.frame_limit,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No analysis rows produced.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["seq_cls", "seq_name", "frame_idx"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Saved CSV: {out_csv}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

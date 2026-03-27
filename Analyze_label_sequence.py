#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# ============================================================
# Regex
# ============================================================
QP_DIR_RE = re.compile(r"^qp(\d+)$")
COMBO_RE = re.compile(r"^tf8_(\-?\d+)_tf16_(\-?\d+)_bs_(\-?\d+)$")
STEM_RE = re.compile(r"^(?P<seq_name>.+)_t(?P<start>\d{4})_t(?P<end>\d{4})$")
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# ============================================================
# Robust BD-rate
# ============================================================
def _prepare_rd_points(
    df: pd.DataFrame,
    bitrate_col: str,
    psnr_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      psnr sorted ascending
      log(rate)
    """
    x = df[[bitrate_col, psnr_col]].dropna().copy()
    x = x[x[bitrate_col] > 0].copy()
    x = x.sort_values(psnr_col)

    psnr = x[psnr_col].to_numpy(dtype=np.float64)
    rate = x[bitrate_col].to_numpy(dtype=np.float64)

    if len(psnr) < 2:
        raise ValueError(f"Need at least 2 valid RD points for {psnr_col}")

    # remove duplicate PSNR
    keep = np.ones(len(psnr), dtype=bool)
    for i in range(1, len(psnr)):
        if abs(psnr[i] - psnr[i - 1]) <= 1e-12:
            keep[i] = False
    psnr = psnr[keep]
    rate = rate[keep]

    if len(psnr) < 2:
        raise ValueError(f"Need at least 2 unique PSNR points for {psnr_col}")

    return psnr, np.log(rate)


def bd_rate_piecewise(
    df_anchor: pd.DataFrame,
    df_test: pd.DataFrame,
    bitrate_col: str,
    psnr_col: str,
) -> float:
    """
    Piecewise-linear BD-rate (%).
    Negative is better.
    """
    p1, lr1 = _prepare_rd_points(df_anchor, bitrate_col, psnr_col)
    p2, lr2 = _prepare_rd_points(df_test, bitrate_col, psnr_col)

    p_min = max(p1.min(), p2.min())
    p_max = min(p1.max(), p2.max())

    if p_max <= p_min:
        raise ValueError(f"No overlapping PSNR range for {psnr_col}")

    xs = np.union1d(p1, p2)
    xs = xs[(xs >= p_min) & (xs <= p_max)]
    xs = np.unique(np.concatenate([xs, [p_min, p_max]]))
    xs.sort()

    y1 = np.interp(xs, p1, lr1)
    y2 = np.interp(xs, p2, lr2)

    int1 = np.trapz(y1, xs)
    int2 = np.trapz(y2, xs)

    avg_diff = (int2 - int1) / (p_max - p_min)
    return (math.exp(avg_diff) - 1.0) * 100.0


# ============================================================
# VTM log parsing
# ============================================================
def parse_vtm_log_summary(log_path: Path) -> Dict[str, float]:
    """
    Parse a VTM-like log.

    Expected pattern:
      ... line containing 'Total Frames'
      next non-empty line contains numeric summary:
         totalFrames bitrate Y-PSNR U-PSNR V-PSNR ...

    We read the first 5 numeric tokens from that next line:
      [total_frames, bitrate, y_psnr, u_psnr, v_psnr]
    """
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for i, line in enumerate(lines):
        if "Total Frames" in line:
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s:
                    nums = NUM_RE.findall(s)
                    if len(nums) >= 5:
                        return {
                            "total_frames": float(nums[0]),
                            "bitrate": float(nums[1]),
                            "y_psnr": float(nums[2]),
                            "u_psnr": float(nums[3]),
                            "v_psnr": float(nums[4]),
                        }
                    break
                j += 1

    raise ValueError(f"Could not find VTM summary in log: {log_path}")


# ============================================================
# Path parsing
# ============================================================
def parse_log_path_info(log_path: Path, codec_root: Path) -> Dict[str, Any]:
    """
    Expected path:
      codec_root/
        qp22/
          tf8_1_tf16_2_bs_16/
            log/ClassA/SeqA_t0000_t0063.log
    """
    rel = log_path.relative_to(codec_root)
    parts = rel.parts

    if len(parts) < 5:
        raise ValueError(f"Unexpected log path structure: {log_path}")

    qp_dir = parts[0]
    combo_dir = parts[1]
    log_marker = parts[2]
    seq_cls = parts[3]
    filename = parts[4]

    if log_marker != "log":
        raise ValueError(f"Expected 'log' folder in path: {log_path}")

    m_qp = QP_DIR_RE.match(qp_dir)
    if m_qp is None:
        raise ValueError(f"Unexpected qp dir: {qp_dir}")

    m_combo = COMBO_RE.match(combo_dir)
    if m_combo is None:
        raise ValueError(f"Unexpected combo dir: {combo_dir}")

    stem = Path(filename).stem
    m_stem = STEM_RE.match(stem)
    if m_stem is None:
        raise ValueError(f"Unexpected log filename stem: {stem}")

    return {
        "qp": int(m_qp.group(1)),
        "tf8": int(m_combo.group(1)),
        "tf16": int(m_combo.group(2)),
        "block_size": int(m_combo.group(3)),
        "seq_cls": seq_cls,
        "seq_name": m_stem.group("seq_name"),
        "start": int(m_stem.group("start")),
        "end": int(m_stem.group("end")),
    }


# ============================================================
# Scan logs -> RD table
# ============================================================
def collect_rd_points(codec_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    log_paths = sorted(codec_root.glob("qp*/tf8_*_tf16_*_bs_*/log/*/*.log"))
    if not log_paths:
        raise RuntimeError(f"No logs found under: {codec_root}")

    for log_path in log_paths:
        try:
            info = parse_log_path_info(log_path, codec_root)
            summ = parse_vtm_log_summary(log_path)

            row = {
                "log_path": str(log_path),
                **info,
                **summ,
            }
            rows.append(row)

        except Exception as e:
            print(f"[WARN] failed to parse log: {log_path} ({e})")

    if not rows:
        raise RuntimeError("No valid log rows parsed.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["seq_cls", "seq_name", "start", "end", "tf8", "tf16", "block_size", "qp"]
    ).reset_index(drop=True)
    return df


# ============================================================
# BD-rate calculation
# ============================================================
def check_required_qps(df: pd.DataFrame, expected_qps: Optional[List[int]]) -> pd.DataFrame:
    if expected_qps is None:
        return df

    expected_qps = sorted(expected_qps)

    good_groups = []
    group_cols = ["seq_cls", "seq_name", "start", "end", "tf8", "tf16", "block_size"]

    for key, g in df.groupby(group_cols):
        got = sorted(g["qp"].astype(int).tolist())
        if got == expected_qps:
            good_groups.append(key)

    keep_mask = np.zeros(len(df), dtype=bool)
    good_set = set(good_groups)

    for idx, row in df.iterrows():
        key = (
            row["seq_cls"], row["seq_name"], int(row["start"]), int(row["end"]),
            int(row["tf8"]), int(row["tf16"]), int(row["block_size"])
        )
        if key in good_set:
            keep_mask[idx] = True

    out = df[keep_mask].copy().reset_index(drop=True)
    return out


def compute_bdrate_against_baseline(
    rd_df: pd.DataFrame,
    baseline_tf8: int,
    baseline_tf16: int,
    baseline_bs: int,
) -> pd.DataFrame:
    """
    Compute BD-rate per:
      (seq_cls, seq_name, start, end, tf8, tf16, block_size)
    against baseline combo:
      (baseline_tf8, baseline_tf16, baseline_bs)
    within same (seq_cls, seq_name, start, end)
    """
    rows = []

    clip_cols = ["seq_cls", "seq_name", "start", "end"]
    combo_cols = ["tf8", "tf16", "block_size"]

    for clip_key, clip_df in rd_df.groupby(clip_cols):
        base_df = clip_df[
            (clip_df["tf8"] == baseline_tf8) &
            (clip_df["tf16"] == baseline_tf16) &
            (clip_df["block_size"] == baseline_bs)
        ].copy()

        if len(base_df) < 2:
            print(f"[WARN] missing baseline RD points for clip={clip_key}")
            continue

        for combo_key, cand_df in clip_df.groupby(combo_cols):
            tf8, tf16, bs = combo_key

            if len(cand_df) < 2:
                continue

            try:
                y_bdr = bd_rate_piecewise(
                    df_anchor=base_df,
                    df_test=cand_df,
                    bitrate_col="bitrate",
                    psnr_col="y_psnr",
                )
            except Exception as e:
                print(f"[WARN] Y BD-rate failed clip={clip_key} combo={combo_key}: {e}")
                y_bdr = np.nan

            try:
                u_bdr = bd_rate_piecewise(
                    df_anchor=base_df,
                    df_test=cand_df,
                    bitrate_col="bitrate",
                    psnr_col="u_psnr",
                )
            except Exception as e:
                print(f"[WARN] U BD-rate failed clip={clip_key} combo={combo_key}: {e}")
                u_bdr = np.nan

            try:
                v_bdr = bd_rate_piecewise(
                    df_anchor=base_df,
                    df_test=cand_df,
                    bitrate_col="bitrate",
                    psnr_col="v_psnr",
                )
            except Exception as e:
                print(f"[WARN] V BD-rate failed clip={clip_key} combo={combo_key}: {e}")
                v_bdr = np.nan

            row = {
                "seq_cls": clip_key[0],
                "seq_name": clip_key[1],
                "start": clip_key[2],
                "end": clip_key[3],
                "tf8": tf8,
                "tf16": tf16,
                "block_size": bs,
                "baseline_tf8": baseline_tf8,
                "baseline_tf16": baseline_tf16,
                "baseline_block_size": baseline_bs,
                "num_rd_points": len(cand_df),
                "y_bdrate": y_bdr,
                "u_bdrate": u_bdr,
                "v_bdrate": v_bdr,
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No BD-rate rows computed.")

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["seq_cls", "seq_name", "start", "end", "tf8", "tf16", "block_size"]
    ).reset_index(drop=True)
    return out


def summarize_combo_mean(bdr_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tf8", "tf16", "block_size"]

    agg = (
        bdr_df.groupby(group_cols, dropna=False)
        .agg(
            count=("y_bdrate", "size"),
            y_bdrate_mean=("y_bdrate", "mean"),
            y_bdrate_median=("y_bdrate", "median"),
            u_bdrate_mean=("u_bdrate", "mean"),
            u_bdrate_median=("u_bdrate", "median"),
            v_bdrate_mean=("v_bdrate", "mean"),
            v_bdrate_median=("v_bdrate", "median"),
        )
        .reset_index()
        .sort_values(["y_bdrate_mean", "u_bdrate_mean", "v_bdrate_mean"], ascending=True)
        .reset_index(drop=True)
    )
    return agg


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Parse VTM logs, gather RD points, compute BD-rate per (tf8, tf16, block_size)"
    )
    parser.add_argument("--codec_root", type=str, required=True, help="Root folder containing qp*/tf8_*_tf16_*_bs_*")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save CSV results")

    parser.add_argument("--baseline_tf8", type=int, required=True)
    parser.add_argument("--baseline_tf16", type=int, required=True)
    parser.add_argument("--baseline_block_size", type=int, required=True)

    parser.add_argument(
        "--expected_qps",
        type=str,
        default="22,27,32,37",
        help='Expected QPs per group, e.g. "22,27,32,37". Empty string disables strict check.',
    )

    args = parser.parse_args()

    codec_root = Path(args.codec_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.expected_qps.strip():
        expected_qps = [int(x.strip()) for x in args.expected_qps.split(",") if x.strip()]
        expected_qps = sorted(expected_qps)
    else:
        expected_qps = None

    print(f"[INFO] Scanning logs under: {codec_root}")
    rd_df = collect_rd_points(codec_root)

    rd_csv = out_dir / "rd_points_all.csv"
    rd_df.to_csv(rd_csv, index=False)
    print(f"[INFO] Saved RD points: {rd_csv} ({len(rd_df)} rows)")

    if expected_qps is not None:
        before = len(rd_df)
        rd_df = check_required_qps(rd_df, expected_qps)
        after = len(rd_df)
        print(f"[INFO] Strict QP filter applied: {before} -> {after} rows")

    bdr_df = compute_bdrate_against_baseline(
        rd_df=rd_df,
        baseline_tf8=args.baseline_tf8,
        baseline_tf16=args.baseline_tf16,
        baseline_bs=args.baseline_block_size,
    )

    bdr_csv = out_dir / "bdrate_per_window.csv"
    bdr_df.to_csv(bdr_csv, index=False)
    print(f"[INFO] Saved per-window BD-rate: {bdr_csv} ({len(bdr_df)} rows)")

    combo_mean_df = summarize_combo_mean(bdr_df)
    combo_mean_csv = out_dir / "bdrate_combo_mean.csv"
    combo_mean_df.to_csv(combo_mean_csv, index=False)
    print(f"[INFO] Saved combo mean BD-rate: {combo_mean_csv} ({len(combo_mean_df)} rows)")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

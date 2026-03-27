#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

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


def _to_list_of_ints(x) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        vals = []
        for s in x.split(","):
            s = s.strip()
            if s:
                vals.append(int(s))
        return vals
    return [int(x)]


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

        ra_cfg = _pick(info, ["ra_cfg"], _pick(defaults, ["ra_cfg"], None))
        intra_period = _pick(info, ["intra_period"], _pick(defaults, ["intra_period"], 32))

        tf8_list = _to_list_of_ints(
            _pick(info, ["tf_strength_8_list"], _pick(defaults, ["tf_strength_8_list"], None))
        )
        tf16_list = _to_list_of_ints(
            _pick(info, ["tf_strength_16_list"], _pick(defaults, ["tf_strength_16_list"], None))
        )
        block_size_list = _to_list_of_ints(
            _pick(info, ["block_size_list"], _pick(defaults, ["block_size_list"], None))
        )

        ra_opts_global = defaults.get("ra_opts", {})
        ra_opts_local = info.get("ra_opts", {})
        if ra_opts_global is None:
            ra_opts_global = {}
        if ra_opts_local is None:
            ra_opts_local = {}
        if not isinstance(ra_opts_global, dict) or not isinstance(ra_opts_local, dict):
            raise ValueError(f"ra_opts must be dict for seq={seq_name}")

        ra_opts = dict(ra_opts_global)
        ra_opts.update(ra_opts_local)

        items.append({
            "name": str(seq_name),
            "seq_cls": str(seq_cls),
            "yuv_path": str(yuv_path),
            "width": int(width),
            "height": int(height),
            "frames": int(frames),
            "frame_rate": float(fps),
            "bit_depth": int(bit_depth),
            "ra_cfg": str(ra_cfg) if ra_cfg is not None else "",
            "intra_period": int(intra_period),
            "tf_strength_8_list": tf8_list,
            "tf_strength_16_list": tf16_list,
            "block_size_list": block_size_list,
            "ra_opts": ra_opts,
        })

    if not items:
        raise RuntimeError("No valid sequences found in YAML after filters.")

    return items


# ============================================================
# Utility
# ============================================================
def parse_int_list_arg(s: str) -> Optional[List[int]]:
    s = s.strip()
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out if out else None


def parse_qp_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("Empty --qps")
    return vals


def parse_scale_list(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("Empty --scales")
    return vals


def build_extra_opts_string(extra_opts: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in extra_opts.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = 1 if v else 0
        parts.append(f"--{k}={shlex.quote(str(v))}")
    return " ".join(parts)


def scaled_even_size(width: int, height: int, scale: float) -> Tuple[int, int]:
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    if new_w % 2 == 1:
        new_w -= 1
    if new_h % 2 == 1:
        new_h -= 1
    new_w = max(2, new_w)
    new_h = max(2, new_h)
    return new_w, new_h


# ============================================================
# Streaming YUV resize
# ============================================================
def get_sample_info(bit_depth: int):
    if bit_depth <= 8:
        return np.uint8, 1
    return np.dtype("<u2"), 2


def stream_resize_yuv420(
    in_path: Path,
    out_path: Path,
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    num_frames: int,
    bit_depth: int,
):
    """
    Streaming frame-by-frame resize for yuv420 planar.
    Saves original too when out_w/out_h == in_w/in_h.
    """
    dtype_np, bytes_per_sample = get_sample_info(bit_depth)

    in_w2, in_h2 = in_w // 2, in_h // 2
    out_w2, out_h2 = out_w // 2, out_h // 2

    y_n = in_w * in_h
    uv_n = in_w2 * in_h2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        for t in range(num_frames):
            yb = fin.read(y_n * bytes_per_sample)
            ub = fin.read(uv_n * bytes_per_sample)
            vb = fin.read(uv_n * bytes_per_sample)

            if len(yb) != y_n * bytes_per_sample or len(ub) != uv_n * bytes_per_sample or len(vb) != uv_n * bytes_per_sample:
                raise IOError(f"EOF while reading {in_path} at frame {t}")

            Y = np.frombuffer(yb, dtype=dtype_np).reshape(in_h, in_w)
            U = np.frombuffer(ub, dtype=dtype_np).reshape(in_h2, in_w2)
            V = np.frombuffer(vb, dtype=dtype_np).reshape(in_h2, in_w2)

            if out_w == in_w and out_h == in_h:
                Yo, Uo, Vo = Y, U, V
            else:
                Yo = cv2.resize(Y, (out_w, out_h), interpolation=cv2.INTER_AREA)
                Uo = cv2.resize(U, (out_w2, out_h2), interpolation=cv2.INTER_AREA)
                Vo = cv2.resize(V, (out_w2, out_h2), interpolation=cv2.INTER_AREA)

            if bit_depth <= 8:
                Yo.astype(np.uint8, copy=False).tofile(fout)
                Uo.astype(np.uint8, copy=False).tofile(fout)
                Vo.astype(np.uint8, copy=False).tofile(fout)
            else:
                Yo.astype(np.dtype("<u2"), copy=False).tofile(fout)
                Uo.astype(np.dtype("<u2"), copy=False).tofile(fout)
                Vo.astype(np.dtype("<u2"), copy=False).tofile(fout)


# ============================================================
# Encoder command generation
# ============================================================
def build_encoder_cmd(
    encoder_app: Path,
    cfg_path: Path,
    input_yuv: Path,
    bitstream_path: Path,
    recon_path: str,
    log_path: Path,
    width: int,
    height: int,
    frames: int,
    qp: int,
    intra_period: int,
    frame_rate: int,
    input_bit_depth: int,
    tf8_opt_name: str,
    tf16_opt_name: str,
    block_size_opt_name: str,
    tf8: Optional[int],
    tf16: Optional[int],
    block_size: Optional[int],
    extra_ra_opts: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        shlex.quote(str(encoder_app)),
        "-c", shlex.quote(str(cfg_path)),
        "-i", shlex.quote(str(input_yuv)),
        "-b", shlex.quote(str(bitstream_path)),
        "-o", shlex.quote(str(recon_path)),
        "-wdt", str(width),
        "-hgt", str(height),
        "-fr", str(frame_rate),
        "-f", str(frames),
        "-q", str(qp),
        f"--InputBitDepth={input_bit_depth}",
        f"--IntraPeriod={intra_period}",
    ]

    if tf8 is not None and tf8_opt_name:
        parts.append(f"{tf8_opt_name}={tf8}")
    if tf16 is not None and tf16_opt_name:
        parts.append(f"{tf16_opt_name}={tf16}")
    if block_size is not None and block_size_opt_name:
        parts.append(f"{block_size_opt_name}={block_size}")

    if extra_ra_opts:
        extra_str = build_extra_opts_string(extra_ra_opts)
        if extra_str:
            parts.append(extra_str)

    cmd = " ".join(parts) + f" > {shlex.quote(str(log_path))} 2>&1"
    return cmd


def build_scaled_yuv_path(
    scaled_root: Path,
    seq_cls: str,
    seq_name: str,
    scale_pct: int,
    width: int,
    height: int,
) -> Path:
    return scaled_root / seq_cls / f"{seq_name}_s{scale_pct:03d}_{width}x{height}.yuv"


def build_codec_paths(
    codec_root: Path,
    seq_cls: str,
    seq_name: str,
    scale_pct: int,
    width: int,
    height: int,
    qp: int,
    tf8: Optional[int],
    tf16: Optional[int],
    block_size: Optional[int],
) -> Tuple[Path, Path, Path]:
    combo_tag = f"s{scale_pct:03d}_{width}x{height}_tf8_{tf8}_tf16_{tf16}_bs_{block_size}"

    bit_path = codec_root / f"qp{qp:02d}" / combo_tag / "bin" / seq_cls / f"{seq_name}.bin"
    rec_path = codec_root / f"qp{qp:02d}" / combo_tag / "rec" / seq_cls / f"{seq_name}.yuv"
    log_path = codec_root / f"qp{qp:02d}" / combo_tag / "log" / seq_cls / f"{seq_name}.log"

    bit_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return bit_path, rec_path, log_path


def submit_bsub_batch(
    cmds: List[str],
    job_name: str,
    queue: str = "",
    extra_bsub_args: str = "",
    dry_run: bool = False,
):
    if not cmds:
        return

    body = " ".join([f"({cmd})&" for cmd in cmds]) + " wait"

    parts = ["bsub", "-J", job_name]
    if queue:
        parts += ["-q", queue]
    if extra_bsub_args.strip():
        parts += shlex.split(extra_bsub_args)
    parts.append(body)

    print("[BSUB CMD]")
    print(" ".join(shlex.quote(p) for p in parts))

    if dry_run:
        return

    subprocess.run(parts, check=True)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Streaming resize UHD YUVs, save all scales including original, and submit VVC sweeps"
    )

    # YAML / sequence selection
    parser.add_argument("--yaml", type=str, required=True, help="dataset yaml (seq dict)")
    parser.add_argument("--only_seq", type=str, default="", help="comma-separated seq names")

    # UHD + scaling
    parser.add_argument("--scaled_yuv_root", type=str, required=True, help="Root folder for scaled yuv outputs")
    parser.add_argument("--uhd_only", action="store_true", help="Process only 3840x2160 sequences")
    parser.add_argument("--uhd_width", type=int, default=3840)
    parser.add_argument("--uhd_height", type=int, default=2160)
    parser.add_argument("--scales", type=str, default="1.00,0.95,0.90,0.85,0.80", help='e.g. "1.00,0.95,0.90"')
    parser.add_argument("--skip_existing_yuv", action="store_true")

    # Encoder / submit
    parser.add_argument("--bin_dir", type=str, required=True, help="Directory containing EncoderApp")
    parser.add_argument("--encoder_name", type=str, default="EncoderApp")
    parser.add_argument("--codec_root", type=str, required=True, help="Root folder for codec outputs")
    parser.add_argument("--default_cfg_path", type=str, default="", help="Fallback cfg if YAML seq/defaults has no ra_cfg")

    parser.add_argument("--submit_bsub", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8, help="How many encoder commands per bsub submission")
    parser.add_argument("--job_prefix", type=str, default="VVC_SWEEP")
    parser.add_argument("--queue", type=str, default="")
    parser.add_argument("--extra_bsub_args", type=str, default="", help='e.g. \'-n 8 -R "span[hosts=1]"\'')
    parser.add_argument("--dry_run_bsub", action="store_true")
    parser.add_argument("--discard_recon", action="store_true", help="Use /dev/null for reconstruction output")

    # encoding params
    parser.add_argument("--qps", type=str, required=True, help='Comma-separated QPs, e.g. "22,27,32,37"')
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=0,
        help="0: use all frames from yaml. >0: use first N frames for every sequence.",
    )

    # Optional global sweep lists (override YAML if provided)
    parser.add_argument("--tf_strength_8_list", type=str, default="", help='e.g. "0,1,2"')
    parser.add_argument("--tf_strength_16_list", type=str, default="", help='e.g. "0,1,2"')
    parser.add_argument("--block_size_list", type=str, default="", help='e.g. "8,16,32"')

    # Actual encoder option names in your VTM branch
    parser.add_argument("--tf8_opt_name", type=str, default="--TFStrength8")
    parser.add_argument("--tf16_opt_name", type=str, default="--TFStrength16")
    parser.add_argument("--block_size_opt_name", type=str, default="--TFBlockSize")

    args = parser.parse_args()

    only_seq = {s.strip() for s in args.only_seq.split(",") if s.strip()}
    qps = parse_qp_list(args.qps)
    scales = parse_scale_list(args.scales)

    global_tf8_list = parse_int_list_arg(args.tf_strength_8_list)
    global_tf16_list = parse_int_list_arg(args.tf_strength_16_list)
    global_block_size_list = parse_int_list_arg(args.block_size_list)

    seq_items = collect_seq_items_from_yaml(
        Path(args.yaml),
        only_seq=only_seq if only_seq else None,
    )

    encoder_app = Path(args.bin_dir) / args.encoder_name
    if not encoder_app.exists():
        raise FileNotFoundError(f"EncoderApp not found: {encoder_app}")

    default_cfg_path = Path(args.default_cfg_path) if args.default_cfg_path else None
    if default_cfg_path is not None and not default_cfg_path.exists():
        raise FileNotFoundError(f"default cfg not found: {default_cfg_path}")

    scaled_yuv_root = Path(args.scaled_yuv_root)
    scaled_yuv_root.mkdir(parents=True, exist_ok=True)

    codec_root = Path(args.codec_root)
    codec_root.mkdir(parents=True, exist_ok=True)

    pending_cmds: List[str] = []
    submitted_job_count = 0
    total_cmd_count = 0

    for i, item in enumerate(seq_items, 1):
        seq_name = item["name"]
        seq_cls = item["seq_cls"]
        yuv_path = Path(item["yuv_path"])
        width = int(item["width"])
        height = int(item["height"])
        total_frames = int(item["frames"])
        fps = int(round(float(item["frame_rate"])))
        bit_depth = int(item["bit_depth"])
        intra_period = int(item["intra_period"])
        ra_opts = dict(item["ra_opts"])

        if args.uhd_only and (width != args.uhd_width or height != args.uhd_height):
            print(f"[SKIP] {seq_name}: not UHD ({width}x{height})")
            continue

        if not yuv_path.exists():
            print(f"[SKIP] missing yuv: {yuv_path}")
            continue

        cfg_path = Path(item["ra_cfg"]) if item["ra_cfg"] else default_cfg_path
        if cfg_path is None:
            raise ValueError(
                f"No RA cfg for seq={seq_name}. "
                f"Set seq.ra_cfg / defaults.ra_cfg in YAML or use --default_cfg_path"
            )
        if not cfg_path.exists():
            raise FileNotFoundError(f"cfg not found for seq={seq_name}: {cfg_path}")

        used_frames = total_frames if args.frame_limit <= 0 else min(args.frame_limit, total_frames)
        if used_frames <= 0:
            print(f"[SKIP] {seq_name}: used_frames <= 0")
            continue

        tf8_list = global_tf8_list if global_tf8_list is not None else item["tf_strength_8_list"]
        tf16_list = global_tf16_list if global_tf16_list is not None else item["tf_strength_16_list"]
        block_size_list = global_block_size_list if global_block_size_list is not None else item["block_size_list"]

        if not tf8_list:
            raise ValueError(f"tf_strength_8_list missing for seq={seq_name}")
        if not tf16_list:
            raise ValueError(f"tf_strength_16_list missing for seq={seq_name}")
        if not block_size_list:
            raise ValueError(f"block_size_list missing for seq={seq_name}")

        print(
            f"[{i}/{len(seq_items)}] seq={seq_name} cls={seq_cls} "
            f"frames={used_frames}/{total_frames} "
            f"tf8={tf8_list} tf16={tf16_list} bs={block_size_list}"
        )

        # ----------------------------------------------------
        # 1) make scaled yuvs by streaming resize
        # ----------------------------------------------------
        scaled_versions: List[Tuple[int, int, int, Path]] = []
        for scale in scales:
            scale_pct = int(round(scale * 100.0))
            out_w, out_h = scaled_even_size(width, height, scale)
            out_yuv_path = build_scaled_yuv_path(
                scaled_root=scaled_yuv_root,
                seq_cls=seq_cls,
                seq_name=seq_name,
                scale_pct=scale_pct,
                width=out_w,
                height=out_h,
            )

            scaled_versions.append((scale_pct, out_w, out_h, out_yuv_path))

            if args.skip_existing_yuv and out_yuv_path.exists():
                print(f"[SKIP YUV] exists: {out_yuv_path}")
                continue

            print(
                f"[INFO] Streaming resize/save: {seq_name} "
                f"{width}x{height} -> {out_w}x{out_h} (s{scale_pct:03d})"
            )
            stream_resize_yuv420(
                in_path=yuv_path,
                out_path=out_yuv_path,
                in_w=width,
                in_h=height,
                out_w=out_w,
                out_h=out_h,
                num_frames=used_frames,
                bit_depth=bit_depth,
            )

        # ----------------------------------------------------
        # 2) submit VVC sweeps for every scaled version
        # ----------------------------------------------------
        for scale_pct, out_w, out_h, scaled_yuv_path in scaled_versions:
            for qp, tf8, tf16, block_size in itertools.product(
                qps, tf8_list, tf16_list, block_size_list
            ):
                bit_path, rec_path, log_path = build_codec_paths(
                    codec_root=codec_root,
                    seq_cls=seq_cls,
                    seq_name=seq_name,
                    scale_pct=scale_pct,
                    width=out_w,
                    height=out_h,
                    qp=qp,
                    tf8=tf8,
                    tf16=tf16,
                    block_size=block_size,
                )

                recon_path = "/dev/null" if args.discard_recon else str(rec_path)

                cmd = build_encoder_cmd(
                    encoder_app=encoder_app,
                    cfg_path=cfg_path,
                    input_yuv=scaled_yuv_path,
                    bitstream_path=bit_path,
                    recon_path=recon_path,
                    log_path=log_path,
                    width=out_w,
                    height=out_h,
                    frames=used_frames,
                    qp=qp,
                    intra_period=intra_period,
                    frame_rate=fps,
                    input_bit_depth=bit_depth,
                    tf8_opt_name=args.tf8_opt_name,
                    tf16_opt_name=args.tf16_opt_name,
                    block_size_opt_name=args.block_size_opt_name,
                    tf8=tf8,
                    tf16=tf16,
                    block_size=block_size,
                    extra_ra_opts=ra_opts,
                )

                pending_cmds.append(cmd)
                total_cmd_count += 1

                if len(pending_cmds) >= args.batch_size:
                    job_name = f"{args.job_prefix}_{submitted_job_count:04d}"
                    submit_bsub_batch(
                        cmds=pending_cmds,
                        job_name=job_name,
                        queue=args.queue,
                        extra_bsub_args=args.extra_bsub_args,
                        dry_run=(not args.submit_bsub) or args.dry_run_bsub,
                    )
                    submitted_job_count += 1
                    pending_cmds = []

    if pending_cmds:
        job_name = f"{args.job_prefix}_{submitted_job_count:04d}"
        submit_bsub_batch(
            cmds=pending_cmds,
            job_name=job_name,
            queue=args.queue,
            extra_bsub_args=args.extra_bsub_args,
            dry_run=(not args.submit_bsub) or args.dry_run_bsub,
        )
        submitted_job_count += 1

    print(f"[INFO] total encoder commands: {total_cmd_count}")
    if args.submit_bsub and not args.dry_run_bsub:
        print(f"[INFO] submitted {submitted_job_count} bsub jobs")
    else:
        print(f"[INFO] dry-run mode: generated {submitted_job_count} bsub batches")
    print(f"[INFO] scaled yuv root: {scaled_yuv_root}")
    print(f"[INFO] codec outputs root: {codec_root}")


if __name__ == "__main__":
    main()

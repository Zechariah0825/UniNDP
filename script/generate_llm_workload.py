import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple


"""
Generate LLM workloads (decode / prefill) for UNiNDP.

Input format for models follows the style in the user's screenshot:

    # ndec, hdim, nheads, dhead, ff_scale, gqaheads
    model_table['LLAMA3-8B']  = [32, 4096, 32, 128, 3.5, 8]

You can freely extend / modify `MODEL_TABLE` below and then run:

    python script/generate_llm_workload.py \\
        --model LLAMA3-8B \\
        --phase decode \\
        --L 32 \\
        --attn mha

This will write two CSVs into `workload/`:

    llama3_8B_decode_tk32_mha.csv      # full operator-level workload
    llama3_8B_decode_mm_mha.csv        # only the mm operators (GEMMs)

Similarly for prefill:

    python script/generate_llm_workload.py --model LLAMA3-8B --phase prefill --L 4096 --attn mha
"""


@dataclass
class ModelCfg:
    ndec: int
    hdim: int
    nheads: int
    dhead: int
    ff_scale: float
    gqaheads: int


# -----------------------------------------------------------------------------
# 1. Model table (filled from the notebook screenshot)
# -----------------------------------------------------------------------------

MODEL_TABLE = {
    # GPT family ---------------------------------------------------------------
    # ndec, hdim, nheads, dhead, ff_scale, gqaheads
    "GPT-175B":   ModelCfg(96, 12288, 96, 128, 4.0, 0),
    "GPT2-1.5B":  ModelCfg(48, 1600, 25, 64, 4.0, 0),
    "GPT3-2.7B":  ModelCfg(32, 2560, 32, 80, 4.0, 0),
    "GPT3-6.7B":  ModelCfg(32, 4096, 32, 128, 4.0, 0),
    "GPT3-13B":   ModelCfg(40, 5120, 40, 128, 4.0, 0),

    # Qwen3 family -------------------------------------------------------------
    "Qwen3-0.6B": ModelCfg(28, 1024, 16, 128, 3.0, 8),
    "Qwen3-1.7B": ModelCfg(28, 2048, 16, 128, 3.0, 8),
    "Qwen3-4B":   ModelCfg(36, 2560, 32, 128, 3.8, 8),
    "Qwen3-8B":   ModelCfg(36, 4096, 32, 128, 3.0, 8),
    "Qwen3-14B":  ModelCfg(40, 5120, 40, 128, 3.4, 8),
    "Qwen3-32B":  ModelCfg(64, 5120, 64, 128, 5.0, 8),

    # LLaMA family -------------------------------------------------------------
    "LLAMA1-7B":  ModelCfg(32, 4096, 32, 128, 2.6875, 0),
    "LLAMA2-7B":  ModelCfg(32, 4096, 32, 128, 2.6875, 0),
    "LLAMA3-8B":  ModelCfg(32, 4096, 32, 128, 3.5, 8),
    "LLAMA3-70B": ModelCfg(80, 8192, 64, 128, 3.5, 8),
}


Row = Tuple[str, str, int, int, int, int]


def _ff_dim(hdim: int, ff_scale: float) -> int:
    # Round to int in case ff_scale is given as float like 2.6875.
    return int(round(hdim * ff_scale))


def build_decode_rows(
    cfg: ModelCfg,
    L: int,
    attn_mode: str,
    vocab_size: int,
    batch_size: int,
) -> List[Row]:
    """
    Single-token decode step.

    Symbols:
        hdim     = d_model
        nheads   = number of query heads
        dhead    = per-head dimension, hdim = nheads * dhead
        ff_dim   = hdim * ff_scale
        L        = cached context length
        kv_heads = gqaheads if GQA else nheads
    """
    hdim = cfg.hdim
    nheads = cfg.nheads
    dhead = cfg.dhead
    ff_dim = _ff_dim(hdim, cfg.ff_scale)
    batch=batch_size
    # If gqaheads is 0, we treat it as "no GQA info" and fall back to MHA.
    use_gqa = attn_mode == "gqa" and cfg.gqaheads > 0
    kv_heads = cfg.gqaheads if use_gqa else nheads
    groups = nheads // kv_heads if use_gqa else 1  # number of query groups

    rows: List[Row] = []

    # QKV projections
    rows.append(("q_gen", "mm", 1, hdim, hdim, batch))
    rows.append(("k_gen", "mm", 1, hdim, hdim, batch))
    rows.append(("v_gen", "mm", 1, hdim, hdim, batch))

    # Per-head embedding scale / bias
    rows.append(("q_emb_mul", "elewise", 1, nheads, 1, dhead*batch))
    rows.append(("q_emb_add", "elewise", 1, nheads, 1, dhead*batch))
    rows.append(("k_emb_mul", "elewise", 1, nheads, 1, dhead*batch))
    rows.append(("k_emb_add", "elewise", 1, nheads, 1, dhead*batch))

    # Attention: QK, softmax, KV
    if use_gqa:
        # GQA flattening (single-token decode)
        # Q  : (groups, d_model / groups)
        # K  : (d_model / groups, L)
        # S  : (groups, L)
        group_dim = dhead * kv_heads  # = d_model / groups
        rows.append(("qk", "mm", groups, group_dim, L, batch))
        # Softmax over length-L scores for each group (approximate encoding)
        rows.append(("softmax", "softmax", groups, L, 1, batch))
        # KV: S (groups, nheads * L)  @  V (nheads * L, dhead)
        rows.append(("kv", "mm", groups, nheads * L, dhead, batch))
    else:
        # MHA flattening (single-token decode)
        # Q  : (1, d_model)
        # K  : (d_model, L)
        # S  : (1, L)
        rows.append(("qk", "mm", 1, hdim, L, batch))
        # Softmax per head over length-L (approximate encoding)
        rows.append(("softmax", "softmax", L, nheads, 1, batch))
        # KV: S (1, nheads * L)  @  V (nheads * L, dhead)
        rows.append(("kv", "mm", 1, nheads * L, dhead, batch))

    # Output projection from attention
    rows.append(("out_proj", "mm", 1, hdim, hdim, batch))
    rows.append(("out_add", "elewise", hdim, 1, 1, batch))
    rows.append(("out_rms_norm", "elewise", hdim, 1, 1, batch))

    # FFN (SwiGLU-style: up, gate, down)
    rows.append(("up", "mm", 1, hdim, ff_dim, batch))
    rows.append(("gate", "mm", 1, hdim, ff_dim, batch))
    rows.append(("up_add", "elewise", ff_dim, 1, 1, batch))
    rows.append(("down", "mm", 1, ff_dim, hdim, batch))
    rows.append(("down_add", "elewise", hdim, 1, 1, batch))

    # Final layer norm + lm head projection
    rows.append(("final_rms_norm", "elewise", hdim, 1, 1, batch))
    rows.append(("final_proj", "mm", 1, hdim, vocab_size, batch))

    return rows


def build_prefill_rows(
    cfg: ModelCfg,
    L: int,
    attn_mode: str,
    vocab_size: int,
    batch_size: int,
) -> List[Row]:
    """
    Prefill (prompt) phase, sequence length = L.
    """
    hdim = cfg.hdim
    nheads = cfg.nheads
    dhead = cfg.dhead
    ff_dim = _ff_dim(hdim, cfg.ff_scale)
    batch=batch_size
    use_gqa = attn_mode == "gqa" and cfg.gqaheads > 0
    kv_heads = cfg.gqaheads if use_gqa else nheads
    groups = nheads // kv_heads if use_gqa else 1

    rows: List[Row] = []

    # QKV projections: (L, hdim, hdim, 1)
    rows.append(("q_gen", "mm", L, hdim, hdim, batch))
    rows.append(("k_gen", "mm", L, hdim, hdim, batch))
    rows.append(("v_gen", "mm", L, hdim, hdim, batch))

    # Per-head embedding scale / bias
    rows.append(("q_emb_mul", "elewise", L, nheads, 1, dhead*batch))
    rows.append(("q_emb_add", "elewise", L, nheads, 1, dhead*batch))
    rows.append(("k_emb_mul", "elewise", L, nheads, 1, dhead*batch))
    rows.append(("k_emb_add", "elewise", L, nheads, 1, dhead*batch))

    # Attention: QK, softmax, KV
    if use_gqa:
        # GQA prefill
        # Q  : (L * groups, d_model / groups)
        # K  : (d_model / groups, L)
        # S  : (L * groups, L)
        group_dim = dhead * kv_heads  # = d_model / groups
        rows.append(("qk", "mm", L * groups, group_dim, L, batch))
        # Softmax per sequence position within each group
        rows.append(("softmax", "softmax", L * groups, L, 1, batch))
        # KV: S (L * groups, nheads * L)  @  V (nheads * L, dhead)
        rows.append(("kv", "mm", L * groups, nheads * L, dhead, batch))
    else:
        # MHA prefill (original llama2-style shapes)
        rows.append(("qk", "mm", L, hdim, L, batch))
        rows.append(("softmax", "softmax", L, kv_heads, L, batch))
        rows.append(("kv", "mm", dhead, kv_heads * L, hdim, batch))

    # Output projection from attention
    rows.append(("out_proj", "mm", L, hdim, hdim, batch))
    rows.append(("out_add", "elewise", L, hdim, 1, batch))
    rows.append(("out_rms_norm", "elewise", L, hdim, 1, batch))

    # FFN
    rows.append(("up", "mm", L, hdim, ff_dim, batch))
    rows.append(("gate", "mm", L, hdim, ff_dim, batch))
    rows.append(("up_add", "elewise", ff_dim, L, 1, batch))
    rows.append(("down", "mm", L, ff_dim, hdim, batch))
    rows.append(("down_add", "elewise", L, hdim, 1, batch))

    # Final norm + projection (only one token projected, following llama2 CSVs)
    rows.append(("final_rms_norm", "elewise", L, 1, 1, batch))
    rows.append(("final_proj", "mm", 1, hdim, vocab_size, batch))

    return rows


def write_workload_csv(
    path: str,
    rows: List[Row],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Keep a blank first line for compatibility with existing workloads.
        writer.writerow([])
        for name, op_type, M, K, N, B in rows:
            writer.writerow([name, op_type, M, K, N, B])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM workloads (decode / prefill).")

    # Exactly one of: built-in table / config CSV / manual parameters.
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model",
        help="Model key, e.g. 'LLAMA3-8B'. Must exist in MODEL_TABLE.",
    )
    mode_group.add_argument(
        "--model-csv",
        help=(
            "Path to a CSV listing models and hyper-parameters. "
            "Header row is required and will be ignored. "
            "Expected columns: name,ndec,hdim,nheads,dhead,ff_scale,gqaheads,tklen."
        ),
    )
    mode_group.add_argument(
        "--manual-name",
        help="Model name when providing all hyper-parameters manually.",
    )
    parser.add_argument(
        "--phase",
        choices=["decode", "prefill"],
        default="decode",
        help="Which phase to generate.",
    )
    parser.add_argument(
        "--tklen",
        type=int,
        default=32,
        help=(
            "Context length tklen (L). For decode this is cached tokens; "
            "for prefill it is the active sequence length. Default: 32."
        ),
    )
    # Backward-compatible alias; if specified it overrides tklen.
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Deprecated alias for --tklen.",
    )
    parser.add_argument(
        "--attn",
        choices=["mha", "gqa", "auto"],
        default="auto",
        help=(
            "Attention layout: 'mha', 'gqa', or 'auto'. "
            "With 'auto' (default), the script infers MHA/GQA from gqaheads: "
            "if 0 or equal to nheads → mha, otherwise → gqa."
        ),
    )
    parser.add_argument(
        "--vocab",
        type=int,
        default=32000,
        help="Vocabulary size V used for final projection.",
    )
    parser.add_argument(
        "--workload-dir",
        default="workload",
        help="Directory to place generated CSV files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size B, default is 1.",
    )
    # Manual hyper-parameters (used only with --manual-name)
    parser.add_argument("--ndec", type=int, help="Decoder layer count.")
    parser.add_argument("--hdim", type=int, help="Model width d_model.")
    parser.add_argument("--nheads", type=int, help="Number of Q heads.")
    parser.add_argument("--dhead", type=int, help="Per-head dimension.")
    parser.add_argument(
        "--ff-scale",
        type=float,
        help="FFN expansion factor; FFN dim = int(round(hdim * ff_scale)).",
    )
    parser.add_argument(
        "--gqaheads",
        type=int,
        help="Number of KV heads for GQA (0 or omit for pure MHA).",
    )

    return parser.parse_args()


def _normalize_prefix(model_name: str) -> str:
    """Convert model name like 'Qwen3-8B' to 'qwen3_8B' for filenames."""
    if "-" in model_name:
        family, size = model_name.split("-", 1)
        return f"{family.lower()}_{size}"
    return model_name.lower()


def generate_for_model(
    model_name: str,
    cfg: ModelCfg,
    tklen: int,
    phase: str,
    attn: str,
    vocab_size: int,
    workload_dir: str,
    batch_size: int,
) -> None:
    """Generate workloads (E2E + mm-only) for a single model."""
    L = tklen

    # Infer attention mode if requested.
    if attn == "auto":
        if cfg.gqaheads and 0 < cfg.gqaheads < cfg.nheads:
            attn_mode = "gqa"
        else:
            attn_mode = "mha"
    else:
        attn_mode = attn

    if phase == "decode":
        rows = build_decode_rows(cfg, L, attn_mode, vocab_size, batch_size)
    else:
        rows = build_prefill_rows(cfg, L, attn_mode, vocab_size, batch_size)

    prefix = _normalize_prefix(model_name)

    # 1) Full operator-level workload (E2E decode / prefill for one step)
    etoe_name = f"{prefix}_{phase}_tk{L}_{attn_mode}.csv"
    etoe_path = os.path.join(workload_dir, etoe_name)
    write_workload_csv(etoe_path, rows)

    # 2) MM-only workload
    mm_rows = [row for row in rows if row[1] == "mm"]
    mm_name = f"{prefix}_{phase}_mm_{attn_mode}.csv"
    mm_path = os.path.join(workload_dir, mm_name)
    write_workload_csv(mm_path, mm_rows)

    print(f"[OK] {model_name}: wrote workload {etoe_path}")
    print(f"[OK] {model_name}: wrote MM-only {mm_path}")


def _read_int(row: dict, key: str, default: int | None = None) -> int:
    val = row.get(key)
    if val is None or val == "":
        if default is None:
            raise ValueError(f"Missing required column '{key}' in CSV row: {row}")
        return default
    return int(val)


def _read_float(row: dict, key: str) -> float:
    val = row.get(key)
    if val is None or val == "":
        raise ValueError(f"Missing required column '{key}' in CSV row: {row}")
    return float(val)


def run_from_csv(
    csv_path: str,
    phase: str,
    attn: str,
    vocab_size: int,
    workload_dir: str,
) -> None:
    """
    Read a CSV of models and hyper-parameters and generate workloads for each.

    Expected header (case-insensitive, order not important):
        name, ndec, hdim, nheads, dhead, ff_scale, gqaheads, tklen
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get("name") or row.get("model") or row.get("Model")
            if not model_name:
                # Skip malformed rows silently.
                continue

            ndec = _read_int(row, "ndec")
            hdim = _read_int(row, "hdim")
            nheads = _read_int(row, "nheads")
            dhead = _read_int(row, "dhead")
            ff_scale = _read_float(row, "ff_scale")
            gqaheads = _read_int(row, "gqaheads", default=0)
            tklen = _read_int(row, "tklen", default=32)

            cfg = ModelCfg(ndec, hdim, nheads, dhead, ff_scale, gqaheads)
            generate_for_model(model_name, cfg, tklen, phase, attn, vocab_size, workload_dir)


def main() -> None:
    args = parse_args()
    tklen = args.L if args.L is not None else args.tklen

    if args.model_csv:
        run_from_csv(args.model_csv, args.phase, args.attn, args.vocab, args.workload_dir)
        return

    if args.manual_name:
        # Ensure manual hyper-parameters are all provided.
        missing = [
            name
            for name, val in [
                ("ndec", args.ndec),
                ("hdim", args.hdim),
                ("nheads", args.nheads),
                ("dhead", args.dhead),
                ("ff_scale", args.ff_scale),
                ("gqaheads", 0 if args.gqaheads is None else args.gqaheads),
            ]
            if val is None and name != "gqaheads"
        ]
        if missing:
            raise SystemExit(f"--manual-name requires hyper-parameters: {', '.join(missing)}")

        cfg = ModelCfg(
            ndec=args.ndec,
            hdim=args.hdim,
            nheads=args.nheads,
            dhead=args.dhead,
            ff_scale=args.ff_scale,
            gqaheads=0 if args.gqaheads is None else args.gqaheads,
        )
        generate_for_model(args.manual_name, cfg, tklen, args.phase, args.attn, args.vocab, args.workload_dir)
        return

    # Default path: look up in built-in MODEL_TABLE
    if args.model not in MODEL_TABLE:
        available = ", ".join(sorted(MODEL_TABLE.keys())) or "<empty – please edit MODEL_TABLE>"
        raise SystemExit(f"Unknown model '{args.model}'. Available keys: {available}")

    cfg = MODEL_TABLE[args.model]
    generate_for_model(args.model, cfg, tklen, args.phase, args.attn, args.vocab, args.workload_dir)


if __name__ == "__main__":
    main()



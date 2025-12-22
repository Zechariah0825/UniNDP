import argparse
import csv
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.pim_utilization import PIMUtilizationAnalyzer


def parse_models_csv(models_csv_path: str, target_model: Optional[str] = None) -> Iterable[Dict[str, float]]:
    with open(models_csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["name"]
            if target_model is not None and name != target_model:
                continue
            yield {
                "name": name,
                "ndec": int(row["ndec"]),
                "hdim": int(row["hdim"]),
                "nheads": int(row["nheads"]),
                "dhead": int(row["dhead"]),
                "ff_scale": float(row["ff_scale"]),
                "gqaheads": int(row.get("gqaheads", 0)),
                "tklen_input": int(row.get("tkleninput", row.get("tklen_input", 32))),
                "tklen_output": int(row.get("tklenoutput", row.get("tklen_output", 32))),
            }


def decode_mm_shapes(
    hdim: int,
    nheads: int,
    dhead: int,
    ff_scale: float,
    gqaheads: Optional[int],
    context_len: int,
    batch: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    ff_dim = int(round(hdim * ff_scale))
    use_gqa = gqaheads is not None and gqaheads > 0 and gqaheads < nheads
    kv_heads = gqaheads if use_gqa else nheads
    groups = nheads // kv_heads if use_gqa else 1

    shapes: Dict[str, Tuple[int, int, int, int]] = {}
    shapes["qkv_gen"] = (1, hdim, 3 * hdim, batch)

    if use_gqa:
        group_dim = dhead * kv_heads
        shapes["qk"] = (groups, group_dim, context_len, batch)
        shapes["kv"] = (groups, nheads * context_len, dhead, batch)
    else:
        shapes["qk"] = (1, hdim, context_len, batch)
        shapes["kv"] = (1, nheads * context_len, dhead, batch)

    shapes["out_proj"] = (1, hdim, hdim, batch)
    shapes["up"] = (1, hdim, ff_dim, batch)
    shapes["down"] = (1, ff_dim, hdim, batch)
    return shapes


def compute_bank_spans(total_units: int, batch: int) -> List[Tuple[int, int]]:
    if batch <= 0 or total_units <= 0:
        return [(0, 0) for _ in range(max(0, batch))]
    base, extra = divmod(total_units, batch)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for idx in range(batch):
        width = base + (1 if idx < extra else 0)
        spans.append((cursor, cursor + width))
        cursor += width
    return spans


def count_effective_units(spans: Sequence[Tuple[int, int]], active_mask: Sequence[bool]) -> int:
    active_units = 0
    for idx, (start, end) in enumerate(spans):
        if idx < len(active_mask) and active_mask[idx]:
            active_units += max(0, end - start)
    return active_units


def parse_tk_out_list(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None:
        return None
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("tk_out entries must be non-negative decode lengths")
        values.append(value)
    if not values:
        raise ValueError("--tk-out-list must contain at least one integer when provided")
    return values


def iter_decode_schedule(
    ctx_start: int,
    ctx_end: int,
    ctx_step: int,
    batch_size: int,
    tk_out_list: Optional[List[int]],
) -> Iterable[Tuple[int, int, List[bool]]]:
    step = max(1, ctx_step)
    if tk_out_list:
        horizon = max(tk_out_list)
        if horizon <= 0:
            return
        for decode_step in range(0, horizon, step):
            context_len = ctx_start + decode_step
            if context_len > ctx_end:
                break
            active_mask = [decode_step < tk_out for tk_out in tk_out_list]
            if not any(active_mask):
                break
            yield context_len, decode_step, active_mask
        return
    for context_len in range(ctx_start, ctx_end + 1, step):
        decode_step = context_len - ctx_start
        yield context_len, decode_step, [True] * batch_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PIM execution utilization for attention decode MMs.")
    parser.add_argument("--models-csv", type=str, default="script/models_llm.csv", help="CSV describing model hyper-parameters.")
    parser.add_argument("--model", type=str, default=None, help="If set, only analyze this model name.")
    parser.add_argument("--architecture", "-A", type=str, default="aim", choices=["aim", "aim8", "hbm-pim", "upmem", "dimmining", "lpddr-pax"], help="Target architecture.")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size during decode.")
    parser.add_argument("--ops", type=str, default="qk,kv", help="Comma-separated list of ops to instrument (subset of qkv_gen,qk,kv,out_proj,up,down).")
    parser.add_argument("--output-csv", type=str, default="pruning_and_breakdown/e2e_detail_pax/pim_decode_util.csv", help="Path to write utilization CSV.")
    parser.add_argument("--po2", action="store_true", help="Require power-of-two partitioning.")
    parser.add_argument("--allow-under-ultize", "-UU", action="store_true", help="Allow under-utilized design points.")
    parser.add_argument("--no-quicksearch", action="store_true", help="Disable predictor-aided search.")
    parser.add_argument("--topk", type=int, default=30, help="Top-K kept during quicksearch.")
    parser.add_argument("--cmdthre", type=float, default=3.0, help="Command-length threshold multiplier.")
    parser.add_argument("--ctx-min", type=int, default=None, help="Override minimum context length. Defaults to tklen_input.")
    parser.add_argument("--ctx-max", type=int, default=None, help="Override maximum context length. Defaults to tklen_output.")
    parser.add_argument("--ctx-step", type=int, default=1, help="Context length sweep step.")
    parser.add_argument(
        "--tk-out-list",
        type=str,
        default=None,
        help=(
            "Comma-separated decode lengths (tokens generated per batch element). "
            "Length must match --batchsize; use 0 to mark slots that finish immediately."
        ),
    )
    parser.add_argument("--print-summary", action="store_true", help="Print utilization summary per op/context.")
    args = parser.parse_args()

    ops = [token.strip() for token in args.ops.split(",") if token.strip()]
    try:
        tk_out_list = parse_tk_out_list(args.tk_out_list)
    except ValueError as exc:
        parser.error(str(exc))
    batch_size = args.batchsize
    if batch_size <= 0:
        parser.error("--batchsize must be a positive integer.")
    if tk_out_list and len(tk_out_list) != batch_size:
        parser.error("Length of --tk-out-list must match --batchsize (use zeros for inactive slots).")
    tk_out_horizon = max(tk_out_list) if tk_out_list else None

    analyzer = PIMUtilizationAnalyzer(
        architecture=args.architecture,
        po2=args.po2,
        allow_under_ultize=args.allow_under_ultize,
        quicksearch=not args.no_quicksearch,
        topk=args.topk,
        cmd_threshold=args.cmdthre,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    layout_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for model_cfg in parse_models_csv(args.models_csv, args.model):
        name = model_cfg["name"]
        hdim = model_cfg["hdim"]
        nheads = model_cfg["nheads"]
        dhead = model_cfg["dhead"]
        ff_scale = model_cfg["ff_scale"]
        gqaheads = model_cfg["gqaheads"]
        ctx_start = args.ctx_min or model_cfg["tklen_input"]
        ctx_end = args.ctx_max or model_cfg["tklen_output"]
        if tk_out_horizon and tk_out_horizon > 0:
            ctx_end = max(ctx_end, ctx_start + tk_out_horizon - 1)

        for context_len, decode_step, active_mask in iter_decode_schedule(
            ctx_start, ctx_end, args.ctx_step, batch_size, tk_out_list
        ):
            active_batches = sum(1 for flag in active_mask if flag)
            inactive_batches = batch_size - active_batches
            shapes = decode_mm_shapes(hdim, nheads, dhead, ff_scale, gqaheads, context_len, batch_size)
            for op in ops:
                if op not in shapes:
                    continue
                cache_key = (args.architecture, name, op)
                state = layout_cache.get(cache_key)
                if state is None:
                    result = analyzer.measure_mm(shapes[op])
                    if result is None:
                        print(f"[warn] Skip {name} op={op} L={context_len}: compilation failed")
                        continue
                    latency, stats, design_point = result
                    total_pim_units = design_point[1]
                    bank_spans = compute_bank_spans(total_pim_units, batch_size)
                    state = {
                        "design_point": design_point,
                        "stats": stats,
                        "latency_ns": latency,
                        "bank_spans": bank_spans,
                        "shape": shapes[op],
                    }
                    layout_cache[cache_key] = state
                else:
                    design_point = state["design_point"]
                    stats = state["stats"]
                    latency = state["latency_ns"]
                    bank_spans = state["bank_spans"]
                    total_pim_units = design_point[1]
                effective_pim_units = count_effective_units(bank_spans, active_mask)
                pim_utilization = (
                    effective_pim_units / total_pim_units if total_pim_units > 0 else 0.0
                )
                active_fraction = active_batches / batch_size if batch_size > 0 else 0.0
                metrics = stats.metrics_with_active_fraction(active_fraction)
                row = {
                    "model": name,
                    "architecture": args.architecture,
                    "batch": batch_size,
                    "context_len": context_len,
                    "decode_step": decode_step,
                    "active_batches": active_batches,
                    "inactive_batches": inactive_batches,
                    "effective_pim_units": effective_pim_units,
                    "total_pim_units": total_pim_units,
                    "pim_utilization": pim_utilization,
                    "op": op,
                    "M": shapes[op][0],
                    "K": shapes[op][1],
                    "N": shapes[op][2],
                    "B": shapes[op][3],
                    "pu_num": design_point[1],
                    "latency_ns": latency,
                }
                row.update(metrics)
                rows.append(row)
                if args.print_summary:
                    util_pct = 100 * pim_utilization
                    print(
                        f"[{name}] step={decode_step} L={context_len} op={op}: latency={latency:.0f}ns, "
                        f"PIM-util={util_pct:.1f}% (active {active_batches}/{batch_size}, "
                        f"banks={effective_pim_units}/{total_pim_units})"
                    )

    if not rows:
        print("No rows generated; nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

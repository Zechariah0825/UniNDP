from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from backend import aim16, aim8, dimmining, hbmpim, upmem
from midend import Mapping, Partition
from sim import sim
from tools import LEVEL, OPTYPE, SimConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

DesignPoint = Tuple[int, int, Sequence, int, Sequence, int, Sequence]


@dataclass
class UtilizationStats:
    sample_count: int = 0
    zero_width_cmds: int = 0
    underutilized_cmds: int = 0
    total_slots: int = 0
    used_slots: int = 0
    sum_util: float = 0.0
    sum_active_width: float = 0.0
    sum_logical_width: float = 0.0
    sum_col_len: float = 0.0
    min_util: float = 1.0
    max_util: float = 0.0
    min_active_width: Optional[int] = None
    max_active_width: Optional[int] = None
    min_logical_width: Optional[int] = None
    max_logical_width: Optional[int] = None
    min_col_len: Optional[int] = None
    max_col_len: Optional[int] = None
    col_len_hist: Dict[int, int] = field(default_factory=dict)
    logical_width_hist: Dict[int, int] = field(default_factory=dict)
    active_width_hist: Dict[int, int] = field(default_factory=dict)
    active_bank_ids: Set[int] = field(default_factory=set)

    def add_sample(
        self,
        logical_width: int,
        active_width: int,
        col_len: int,
        pu_mask: Sequence[int],
    ) -> None:
        if logical_width <= 0 or col_len <= 0:
            return
        util = active_width / logical_width if logical_width else 0.0
        self.sample_count += 1
        self.sum_util += util
        self.total_slots += logical_width * col_len
        self.used_slots += active_width * col_len
        self.sum_active_width += active_width
        self.sum_logical_width += logical_width
        self.sum_col_len += col_len

        self.min_util = min(self.min_util, util)
        self.max_util = max(self.max_util, util)
        self.min_active_width = active_width if self.min_active_width is None else min(
            self.min_active_width, active_width
        )
        self.max_active_width = active_width if self.max_active_width is None else max(
            self.max_active_width, active_width
        )
        self.min_logical_width = logical_width if self.min_logical_width is None else min(
            self.min_logical_width, logical_width
        )
        self.max_logical_width = logical_width if self.max_logical_width is None else max(
            self.max_logical_width, logical_width
        )
        self.min_col_len = col_len if self.min_col_len is None else min(self.min_col_len, col_len)
        self.max_col_len = col_len if self.max_col_len is None else max(self.max_col_len, col_len)

        if active_width < logical_width:
            self.underutilized_cmds += 1
        if active_width == 0:
            self.zero_width_cmds += 1

        self.col_len_hist[col_len] = self.col_len_hist.get(col_len, 0) + 1
        self.logical_width_hist[logical_width] = self.logical_width_hist.get(logical_width, 0) + 1
        self.active_width_hist[active_width] = self.active_width_hist.get(active_width, 0) + 1
        for idx, flag in enumerate(pu_mask):
            if flag:
                self.active_bank_ids.add(idx)

    def metrics(self) -> Dict[str, float]:
        if self.sample_count == 0 or self.total_slots == 0:
            return {
                "inst_samples": 0,
                "avg_util": 0.0,
                "slot_util": 0.0,
                "underutilized_ratio": 0.0,
                "partial_row_ratio": 0.0,
                "zero_width_ratio": 0.0,
                "avg_active_width": 0.0,
                "avg_logical_width": 0.0,
                "avg_col_len": 0.0,
                "min_util": 0.0,
                "max_util": 0.0,
                "min_active_width": 0.0,
                "max_active_width": 0.0,
                "max_logical_width": 0.0,
                "min_col_len": 0.0,
                "max_col_len": 0.0,
                "total_slots": 0,
                "used_slots": 0,
                "underutilized_cmds": 0,
                "partial_row_cmds": 0,
                "zero_width_cmds": 0,
                "unique_active_banks": 0,
            }

        max_col_len = max(self.col_len_hist.keys()) if self.col_len_hist else 0
        partial_cmds = sum(count for length, count in self.col_len_hist.items() if length < max_col_len)

        return {
            "inst_samples": self.sample_count,
            "avg_util": self.sum_util / self.sample_count,
            "slot_util": self.used_slots / self.total_slots if self.total_slots else 0.0,
            "underutilized_ratio": self.underutilized_cmds / self.sample_count,
            "partial_row_ratio": partial_cmds / self.sample_count if self.sample_count else 0.0,
            "zero_width_ratio": self.zero_width_cmds / self.sample_count,
            "avg_active_width": self.sum_active_width / self.sample_count,
            "avg_logical_width": self.sum_logical_width / self.sample_count,
            "avg_col_len": self.sum_col_len / self.sample_count,
            "min_util": self.min_util,
            "max_util": self.max_util,
            "min_active_width": float(self.min_active_width or 0),
            "max_active_width": float(self.max_active_width or 0),
            "min_logical_width": float(self.min_logical_width or 0),
            "max_logical_width": float(self.max_logical_width or 0),
            "min_col_len": float(self.min_col_len or 0),
            "max_col_len": float(max_col_len),
            "total_slots": self.total_slots,
            "used_slots": self.used_slots,
            "underutilized_cmds": self.underutilized_cmds,
            "partial_row_cmds": partial_cmds,
            "zero_width_cmds": self.zero_width_cmds,
            "unique_active_banks": len(self.active_bank_ids),
        }

    def metrics_with_active_fraction(self, active_fraction: float) -> Dict[str, float]:
        """Return metrics adjusted for an active batch fraction without re-running codegen."""

        metrics = self.metrics().copy()
        if not metrics["inst_samples"]:
            metrics["active_fraction"] = 0.0
            metrics["inactive_fraction"] = 1.0
            metrics["effective_slot_util"] = 0.0
            metrics["effective_avg_active_width"] = 0.0
            metrics["effective_used_slots"] = 0.0
            return metrics

        fraction = max(0.0, min(1.0, active_fraction))
        metrics["active_fraction"] = fraction
        metrics["inactive_fraction"] = 1.0 - fraction
        metrics["effective_slot_util"] = metrics["slot_util"] * fraction
        metrics["effective_avg_active_width"] = metrics["avg_active_width"] * fraction
        metrics["effective_used_slots"] = metrics["used_slots"] * fraction
        return metrics


def configure_architecture(architecture: str, workload: str = "mm"):
    arch = architecture.lower()
    if arch == "aim":
        SimConfig.read_from_yaml(str(CONFIG_DIR / "gddr6-aim.yaml"))
        SimConfig.de_pu = [16] if workload == "mm" else [4]
        codegen_cls = aim16
    elif arch == "aim8":
        SimConfig.read_from_yaml(str(CONFIG_DIR / "gddr6-aim.yaml"))
        SimConfig.de_pu = [8] if workload == "mm" else [4]
        codegen_cls = aim8
    elif arch in {"hbm-pim", "lpddr-pax"}:
        config_name = "lpddr-pax.yaml" if arch == "lpddr-pax" else "hbm-pim.yaml"
        SimConfig.read_from_yaml(str(CONFIG_DIR / config_name))
        codegen_cls = hbmpim
    elif arch == "upmem":
        SimConfig.read_from_yaml(str(CONFIG_DIR / "upmem.yaml"))
        codegen_cls = upmem
    elif arch == "dimmining":
        SimConfig.read_from_yaml(str(CONFIG_DIR / "dimmining.yaml"))
        codegen_cls = dimmining
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    SimConfig.pu_level = LEVEL.RA if arch == "dimmining" else LEVEL.DE
    return codegen_cls


def build_design_space(mm_size: Tuple[int, int, int, int], architecture: str, po2: bool, allow_under: bool) -> List[DesignPoint]:
    partition_tool = Partition(require_power_of_2=po2)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered = partition_tool.choose_from_partition_space_mm(partition_space)
    if not allow_under:
        partition_space = filtered

    design_space: List[DesignPoint] = []
    for compute_level, pu_num, partition in partition_space:
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = partition_tool.mem_partition_mm(mm_size, partition)
        for input_choice in reversed(mkl_Input_to_row):
            if architecture in {"aim", "aim8"}:
                if ml_Out_to_row:
                    output_choice = ml_Out_to_row[0]
                    design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
                continue
            for output_choice in reversed(ml_Out_to_row):
                design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
    return design_space


def iter_pu_instructions(gen_code: Sequence[Tuple[int, Sequence[int], Sequence[Tuple]]]):
    for _, _, cmd_list in gen_code:
        for inst in cmd_list:
            if inst[1] != OPTYPE.pu:
                continue
            if inst[0] == LEVEL.DE:
                pu_num, pu_mask = inst[5]
                col_len = inst[8]
            elif inst[0] == LEVEL.RA:
                pu_num, pu_mask = inst[4]
                col_len = inst[7]
            else:
                continue
            yield pu_num, pu_mask, col_len


def collect_pim_stats(gen_code: Sequence[Tuple[int, Sequence[int], Sequence[Tuple]]]) -> UtilizationStats:
    stats = UtilizationStats()
    for logical_width, pu_mask, col_len in iter_pu_instructions(gen_code):
        active_width = sum(1 for flag in pu_mask if flag)
        stats.add_sample(logical_width, active_width, col_len, pu_mask)
    return stats


@dataclass
class CodegenEvaluation:
    design_point: DesignPoint
    gen_code: Optional[Sequence[Tuple]]
    inst_count: Optional[Sequence[int]]
    predict_result: Optional[float]
    latency: Optional[float]


class PIMUtilizationAnalyzer:
    def __init__(
        self,
        architecture: str,
        po2: bool = True,
        allow_under_ultize: bool = False,
        quicksearch: bool = True,
        topk: int = 30,
        cmd_threshold: float = 3.0,
    ) -> None:
        self.architecture = architecture
        self.po2 = po2
        self.allow_under_ultize = allow_under_ultize
        self.quicksearch = quicksearch
        self.topk = topk
        self.cmd_threshold = cmd_threshold
        self.codegen_cls = configure_architecture(architecture, workload="mm")

    def _run_codegen(
        self,
        design_point: DesignPoint,
        cmd_threshold: float,
        need_codegen: bool,
        run_sim: bool,
    ) -> Optional[CodegenEvaluation]:
        (
            compute_level,
            pu_num,
            partition,
            simd_k,
            mkl_Input_to_row,
            simd_l,
            ml_Out_to_row,
        ) = design_point

        mapping_tool = Mapping(require_power_of_2=self.po2)
        hw_id_list = mapping_tool.assign_hw(partition)
        mem_mapping = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition)

        codegen_tool = self.codegen_cls(require_power_of_2=self.po2)
        if need_codegen or run_sim:
            codegen_tool.set_gen()
        gen_code, inst_count, predict_result = codegen_tool.codegen(
            "mm",
            compute_level,
            pu_num,
            partition,
            simd_k,
            mkl_Input_to_row,
            simd_l,
            ml_Out_to_row,
            hw_id_list,
            mem_mapping,
            cmd_threshold=cmd_threshold,
        )
        if gen_code is None:
            return None
        latency = sim(gen_code, silent=True, filename=None) if run_sim else None
        payload = gen_code if need_codegen or run_sim else None
        return CodegenEvaluation(design_point, payload, inst_count, predict_result, latency)

    def _select_candidates(self, design_space: List[DesignPoint]) -> List[DesignPoint]:
        if not self.quicksearch:
            return design_space
        ranked: List[Tuple[float, DesignPoint]] = []
        for design_point in design_space:
            eval_result = self._run_codegen(
                design_point,
                cmd_threshold=self.cmd_threshold,
                need_codegen=False,
                run_sim=False,
            )
            if eval_result is None or eval_result.predict_result is None:
                continue
            ranked.append((eval_result.predict_result, design_point))
            if len(ranked) > self.topk:
                ranked.sort(key=lambda item: item[0])
                ranked = ranked[: self.topk]
        ranked.sort(key=lambda item: item[0])
        if not ranked:
            return design_space
        return [dp for _, dp in ranked]

    def measure_mm(self, mm_shape: Tuple[int, int, int, int]) -> Optional[Tuple[float, UtilizationStats, DesignPoint]]:
        M, K, N, B = mm_shape
        mm_size = (M * B, K, N, 1)
        design_space = build_design_space(mm_size, self.architecture, self.po2, self.allow_under_ultize)
        if not design_space:
            return None
        candidates = self._select_candidates(design_space) if self.quicksearch else design_space
        best_latency = None
        best_stats = None
        best_dp = None
        for design_point in candidates:
            eval_result = self._run_codegen(
                design_point,
                cmd_threshold=0.0,
                need_codegen=True,
                run_sim=True,
            )
            if eval_result is None or eval_result.gen_code is None or eval_result.latency is None:
                continue
            curr_stats = collect_pim_stats(eval_result.gen_code)
            if best_latency is None or eval_result.latency < best_latency:
                best_latency = eval_result.latency
                best_stats = curr_stats
                best_dp = design_point
        if best_latency is None or best_stats is None or best_dp is None:
            return None
        return best_latency, best_stats, best_dp

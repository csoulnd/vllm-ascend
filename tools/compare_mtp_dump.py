#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare MTP forward dumps from two runs (e.g. 910 golden vs 310p).

Each dump file is produced by vllm-ascend when VLLM_ASCEND_MTP_DUMP=1:
  forward_step{step:04d}_{910|310p}.pt

Example:
  python tools/compare_mtp_dump.py \\
    --ref-dir /tmp/mtp_dump/golden \\
    --cmp-dir /tmp/mtp_dump/310p \\
    --ref-tag 910 --cmp-tag 310p

  python tools/compare_mtp_dump.py \\
    --ref-dir /path/to/910_dumps \\
    --cmp-dir /path/to/310p_dumps \\
    --steps 1,2,3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch

DUMP_NAME_RE = re.compile(r"^forward_step(\d+)_(\w+)\.pt$")

INPUT_TENSOR_KEYS = (
    "input_ids",
    "positions",
    "inputs_embeds",
    "logits_indices",
    "query_start_loc",
)

OUTPUT_TENSOR_KEYS = (
    "hidden_states",
    "sample_hidden_states",
)

SPEC_TENSOR_KEYS = (
    "draft_token_ids",
    "target_logits_indices",
    "bonus_logits_indices",
    "logits_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MTP target-model forward dumps between two directories.",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Reference (golden) dump directory, e.g. 910 run.",
    )
    parser.add_argument(
        "--cmp-dir",
        type=Path,
        required=True,
        help="Compare (suspect) dump directory, e.g. 310p run.",
    )
    parser.add_argument(
        "--ref-tag",
        type=str,
        default="910",
        help="Filename tag for reference dumps (default: 910).",
    )
    parser.add_argument(
        "--cmp-tag",
        type=str,
        default="310p",
        help="Filename tag for compare dumps (default: 310p).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help='Steps to compare: "all" or comma-separated ids, e.g. "1,2,3".',
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for float tensor compare (default: 0, exact).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for float tensor compare (default: 0, exact).",
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=5,
        help="When tensors differ, print top-k largest abs diffs (default: 5).",
    )
    return parser.parse_args()


def parse_steps_arg(steps_arg: str) -> set[int] | None:
    if not steps_arg or steps_arg.lower() == "all":
        return None
    return {int(part.strip()) for part in steps_arg.split(",") if part.strip()}


def find_dump_path(directory: Path, step: int, tag: str) -> Path:
    path = directory / f"forward_step{step:04d}_{tag}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dump file: {path}")
    return path


def discover_steps(directory: Path, tag: str) -> list[int]:
    steps: list[int] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = DUMP_NAME_RE.match(path.name)
        if match and match.group(2) == tag:
            steps.append(int(match.group(1)))
    return sorted(steps)


def load_record(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compare_scalar(name: str, ref: Any, cmp: Any, mismatches: list[str]) -> None:
    if ref == cmp:
        return
    mismatches.append(f"{name}: ref={ref!r} cmp={cmp!r}")


def compare_list(name: str, ref: list, cmp: list, mismatches: list[str]) -> None:
    if ref == cmp:
        return
    mismatches.append(f"{name}: ref={ref} cmp={cmp}")


def tensor_stats(ref: torch.Tensor, cmp: torch.Tensor) -> dict[str, Any]:
    ref_f = ref.float()
    cmp_f = cmp.float()
    diff = (ref_f - cmp_f).abs()
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "ref_shape": tuple(ref.shape),
        "cmp_shape": tuple(cmp.shape),
    }


def compare_tensor(
    name: str,
    ref: torch.Tensor | None,
    cmp: torch.Tensor | None,
    rtol: float,
    atol: float,
    show_topk: int,
    mismatches: list[str],
) -> None:
    if ref is None and cmp is None:
        return
    if ref is None or cmp is None:
        mismatches.append(f"{name}: ref_is_none={ref is None} cmp_is_none={cmp is None}")
        return
    if ref.shape != cmp.shape:
        stats = tensor_stats(ref, cmp)
        mismatches.append(f"{name}: shape mismatch {stats}")
        return
    if ref.dtype != cmp.dtype:
        ref = ref.to(cmp.dtype)
    if torch.equal(ref, cmp):
        return
    if ref.is_floating_point() and torch.allclose(ref, cmp, rtol=rtol, atol=atol):
        stats = tensor_stats(ref, cmp)
        mismatches.append(f"{name}: allclose within tol (rtol={rtol}, atol={atol}) {stats}")
        return

    stats = tensor_stats(ref, cmp)
    detail = f"{name}: DIFF {stats}"
    if show_topk > 0 and ref.numel() > 0:
        diff = (ref.float() - cmp.float()).abs().flatten()
        k = min(show_topk, diff.numel())
        topv, topi = torch.topk(diff, k)
        detail += f" top{k}_diff_idx={topi.tolist()} top{k}_diff_val={topv.tolist()}"
        if ref.dim() == 1:
            detail += f" ref_at_idx={ref.flatten()[topi].tolist()} cmp_at_idx={cmp.flatten()[topi].tolist()}"
    mismatches.append(detail)


def compare_record(
    ref_rec: dict[str, Any],
    cmp_rec: dict[str, Any],
    rtol: float,
    atol: float,
    show_topk: int,
) -> list[str]:
    mismatches: list[str] = []

    compare_scalar("step", ref_rec.get("step"), cmp_rec.get("step"), mismatches)
    compare_scalar("path", ref_rec.get("path"), cmp_rec.get("path"), mismatches)
    compare_scalar(
        "num_tokens_unpadded",
        ref_rec.get("num_tokens_unpadded"),
        cmp_rec.get("num_tokens_unpadded"),
        mismatches,
    )
    compare_scalar(
        "num_tokens_padded",
        ref_rec.get("num_tokens_padded"),
        cmp_rec.get("num_tokens_padded"),
        mismatches,
    )
    compare_list("req_ids", ref_rec.get("req_ids", []), cmp_rec.get("req_ids", []), mismatches)

    ref_in = ref_rec.get("inputs") or {}
    cmp_in = cmp_rec.get("inputs") or {}
    compare_list(
        "inputs.per_req_input_ids",
        ref_in.get("per_req_input_ids", []),
        cmp_in.get("per_req_input_ids", []),
        mismatches,
    )
    for key in INPUT_TENSOR_KEYS:
        compare_tensor(
            f"inputs.{key}",
            ref_in.get(key),
            cmp_in.get(key),
            rtol,
            atol,
            show_topk,
            mismatches,
        )

    ref_spec = ref_in.get("spec_decode")
    cmp_spec = cmp_in.get("spec_decode")
    if ref_spec is None and cmp_spec is None:
        pass
    elif ref_spec is None or cmp_spec is None:
        mismatches.append(
            f"inputs.spec_decode: ref_is_none={ref_spec is None} cmp_is_none={cmp_spec is None}"
        )
    else:
        for key in SPEC_TENSOR_KEYS:
            compare_tensor(
                f"inputs.spec_decode.{key}",
                ref_spec.get(key),
                cmp_spec.get(key),
                rtol,
                atol,
                show_topk,
                mismatches,
            )

    ref_out = ref_rec.get("outputs") or {}
    cmp_out = cmp_rec.get("outputs") or {}
    for key in OUTPUT_TENSOR_KEYS:
        compare_tensor(
            f"outputs.{key}",
            ref_out.get(key),
            cmp_out.get(key),
            rtol,
            atol,
            show_topk,
            mismatches,
        )

    return mismatches


def classify_divergence(mismatches: list[str]) -> str:
    input_prefixes = ("inputs.", "num_tokens", "req_ids")
    output_prefixes = ("outputs.",)
    has_input = any(m.startswith(input_prefixes) for m in mismatches)
    has_output = any(m.startswith(output_prefixes) for m in mismatches)
    if has_input and has_output:
        return "inputs_and_outputs"
    if has_input:
        return "inputs_only"
    if has_output:
        return "outputs_only"
    return "metadata_only"


def main() -> int:
    args = parse_args()
    ref_dir = args.ref_dir.expanduser().resolve()
    cmp_dir = args.cmp_dir.expanduser().resolve()
    if not ref_dir.is_dir():
        print(f"ERROR: ref-dir not found: {ref_dir}", file=sys.stderr)
        return 2
    if not cmp_dir.is_dir():
        print(f"ERROR: cmp-dir not found: {cmp_dir}", file=sys.stderr)
        return 2

    step_filter = parse_steps_arg(args.steps)
    ref_steps = discover_steps(ref_dir, args.ref_tag)
    cmp_steps = discover_steps(cmp_dir, args.cmp_tag)
    common_steps = sorted(set(ref_steps) & set(cmp_steps))
    if step_filter is not None:
        common_steps = [s for s in common_steps if s in step_filter]

    if not common_steps:
        print(
            f"ERROR: no common steps. ref={ref_steps} ({args.ref_tag}), "
            f"cmp={cmp_steps} ({args.cmp_tag})",
            file=sys.stderr,
        )
        return 2

    only_ref = sorted(set(ref_steps) - set(cmp_steps))
    only_cmp = sorted(set(cmp_steps) - set(ref_steps))
    if only_ref:
        print(f"WARN: steps only in ref-dir: {only_ref}")
    if only_cmp:
        print(f"WARN: steps only in cmp-dir: {only_cmp}")

    print(f"Comparing {len(common_steps)} step(s): {common_steps}")
    print(f"  ref: {ref_dir} (tag={args.ref_tag})")
    print(f"  cmp: {cmp_dir} (tag={args.cmp_tag})")
    print(f"  tol: rtol={args.rtol} atol={args.atol}")
    print()

    all_ok = True
    first_fail: tuple[int, str] | None = None

    for step in common_steps:
        ref_path = find_dump_path(ref_dir, step, args.ref_tag)
        cmp_path = find_dump_path(cmp_dir, step, args.cmp_tag)
        ref_rec = load_record(ref_path)
        cmp_rec = load_record(cmp_path)
        mismatches = compare_record(ref_rec, cmp_rec, args.rtol, args.atol, args.show_topk)

        if not mismatches:
            print(f"[step {step:04d}] OK — inputs and outputs match")
            continue

        all_ok = False
        kind = classify_divergence(mismatches)
        if first_fail is None:
            first_fail = (step, kind)
        print(f"[step {step:04d}] MISMATCH ({kind}) — {len(mismatches)} issue(s)")
        for line in mismatches:
            print(f"  - {line}")
        print()

    print("=" * 60)
    if all_ok:
        print("RESULT: ALL STEPS MATCH")
        return 0

    assert first_fail is not None
    step, kind = first_fail
    print(f"RESULT: DIVERGED at step {step} ({kind})")
    if kind == "inputs_only":
        print("  -> Check scheduler/token scatter before forward (not GDN yet).")
    elif kind == "outputs_only":
        print("  -> Inputs match; suspect model ops (e.g. GDN/linear_attn) or KV state.")
    elif kind == "inputs_and_outputs":
        print("  -> Both inputs and outputs differ; fix inputs first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

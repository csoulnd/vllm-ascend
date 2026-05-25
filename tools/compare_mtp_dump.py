#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare two MTP forward dump .pt files (e.g. 910 golden vs 310p).

Example:
  python tools/compare_mtp_dump.py \\
    /tmp/mtp_dump/golden/forward_step0002_910.pt \\
    /tmp/mtp_dump/310p/forward_step0002_310p.pt

  python tools/compare_mtp_dump.py ref.pt cmp.pt --print-contents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

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
        description="Compare two MTP forward dump .pt files.",
    )
    parser.add_argument("ref_pt", type=Path, help="Reference (golden) .pt file.")
    parser.add_argument("cmp_pt", type=Path, help="Compare (suspect) .pt file.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for float tensors (default: 0).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for float tensors (default: 0).",
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=5,
        help="When tensors differ, show top-k abs diffs (default: 5).",
    )
    parser.add_argument(
        "--print-contents",
        action="store_true",
        help="Print both .pt records before comparing.",
    )
    parser.add_argument(
        "--full-hidden",
        action="store_true",
        help="With --print-contents, print full hidden_states tensor.",
    )
    parser.add_argument(
        "--max-tensor-elems",
        type=int,
        default=128,
        help="Max elements printed per large tensor (default: 128).",
    )
    return parser.parse_args()


def load_record(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_tensor(
    tensor: torch.Tensor | None,
    *,
    max_elems: int,
    full: bool = False,
) -> str:
    if tensor is None:
        return "None"
    t = tensor.detach().cpu()
    header = f"dtype={t.dtype} shape={tuple(t.shape)} numel={t.numel()}"
    if t.numel() == 0:
        return f"{header} value=[]"
    if full or t.numel() <= max_elems:
        return f"{header}\n  value={t.flatten().tolist()}"
    flat = t.flatten()
    shown = flat[:max_elems].tolist()
    suffix = f" ... (+{t.numel() - max_elems} more)" if t.numel() > max_elems else ""
    return f"{header}\n  value(head)={shown}{suffix}"


def format_record(
    record: dict[str, Any],
    *,
    max_elems: int,
    full_hidden: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"  step={record.get('step')} path={record.get('path')} model={record.get('model')}")
    lines.append(f"  req_ids={record.get('req_ids')}")
    lines.append(
        f"  num_tokens_unpadded={record.get('num_tokens_unpadded')} "
        f"num_tokens_padded={record.get('num_tokens_padded')}"
    )

    inputs = record.get("inputs") or {}
    lines.append("  [inputs]")
    lines.append(f"    per_req_input_ids={inputs.get('per_req_input_ids')}")
    for key in INPUT_TENSOR_KEYS:
        lines.append(f"    {key}:")
        lines.append(f"      {format_tensor(inputs.get(key), max_elems=max_elems)}")

    spec = inputs.get("spec_decode")
    if spec is not None:
        lines.append("    [spec_decode]")
        for key in SPEC_TENSOR_KEYS:
            lines.append(f"      {key}:")
            lines.append(f"        {format_tensor(spec.get(key), max_elems=max_elems)}")

    outputs = record.get("outputs") or {}
    lines.append("  [outputs]")
    for key in OUTPUT_TENSOR_KEYS:
        lines.append(f"    {key}:")
        full = full_hidden and key == "hidden_states"
        lines.append(f"      {format_tensor(outputs.get(key), max_elems=max_elems, full=full)}")

    return "\n".join(lines)


def print_record(label: str, path: Path, record: dict[str, Any], *, max_elems: int, full_hidden: bool) -> None:
    print(f"[{label}] {path}")
    print(format_record(record, max_elems=max_elems, full_hidden=full_hidden))
    print()


def compare_scalar(name: str, ref: Any, cmp: Any, mismatches: list[str]) -> None:
    if ref == cmp:
        return
    mismatches.append(f"{name}: ref={ref!r} cmp={cmp!r}")


def compare_list(name: str, ref: list, cmp: list, mismatches: list[str]) -> None:
    if ref == cmp:
        return
    mismatches.append(f"{name}: ref={ref} cmp={cmp}")


def tensor_stats(ref: torch.Tensor, cmp: torch.Tensor) -> dict[str, Any]:
    diff = (ref.float() - cmp.float()).abs()
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
        mismatches.append(f"{name}: shape mismatch {tensor_stats(ref, cmp)}")
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
        detail += f" top{k}_idx={topi.tolist()} top{k}_val={topv.tolist()}"
        if ref.dim() == 1:
            detail += f" ref={ref.flatten()[topi].tolist()} cmp={cmp.flatten()[topi].tolist()}"
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
    ref_path = args.ref_pt.expanduser().resolve()
    cmp_path = args.cmp_pt.expanduser().resolve()

    if not ref_path.is_file():
        print(f"ERROR: ref file not found: {ref_path}", file=sys.stderr)
        return 2
    if not cmp_path.is_file():
        print(f"ERROR: cmp file not found: {cmp_path}", file=sys.stderr)
        return 2

    ref_rec = load_record(ref_path)
    cmp_rec = load_record(cmp_path)

    print(f"REF: {ref_path}")
    print(f"CMP: {cmp_path}")
    print(f"tol: rtol={args.rtol} atol={args.atol}")
    print()

    if args.print_contents:
        print_record("REF", ref_path, ref_rec, max_elems=args.max_tensor_elems, full_hidden=args.full_hidden)
        print_record("CMP", cmp_path, cmp_rec, max_elems=args.max_tensor_elems, full_hidden=args.full_hidden)

    mismatches = compare_record(ref_rec, cmp_rec, args.rtol, args.atol, args.show_topk)

    if not mismatches:
        print("RESULT: MATCH")
        return 0

    kind = classify_divergence(mismatches)
    print(f"RESULT: MISMATCH ({kind}) — {len(mismatches)} issue(s)")
    for line in mismatches:
        print(f"  - {line}")
    if kind == "inputs_only":
        print("  -> Inputs differ; check scheduler/token layout before forward.")
    elif kind == "outputs_only":
        print("  -> Inputs match; suspect model ops (e.g. GDN) or KV state.")
    elif kind == "inputs_and_outputs":
        print("  -> Both differ; fix inputs first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

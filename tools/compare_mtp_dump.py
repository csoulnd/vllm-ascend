#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare two MTP forward dump .pt files (910 golden vs 310p).

Flow: (1) strict compare on **pre-forward inputs** only; (2) cosine on outputs.
req_ids / step / path / num_tokens are informational — never scored.

Example:
  python tools/compare_mtp_dump.py ref.pt cmp.pt
  python tools/compare_mtp_dump.py ref.pt cmp.pt --cosine-min 0.9999
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Pre-forward tensors (strict, value-equal; dtype may differ).
FORWARD_INPUT_KEYS = (
    "input_ids",
    "positions",
    "logits_indices",
    "query_start_loc",
)

FORWARD_INPUT_OPTIONAL_KEYS = ("inputs_embeds",)

SPEC_INPUT_KEYS = (
    "draft_token_ids",
    "target_logits_indices",
    "bonus_logits_indices",
    "logits_indices",
)

OUTPUT_COSINE_KEYS = (
    "hidden_states",
    "sample_hidden_states",
)

# Never scored — only printed when different.
INFO_KEYS = ("step", "path", "model", "req_ids", "num_tokens_unpadded", "num_tokens_padded")


@dataclass
class FieldResult:
    name: str
    ok: bool
    detail: str


@dataclass
class SectionReport:
    ok: bool = True
    fields: list[FieldResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.fields.append(FieldResult(name, ok, detail))
        if not ok:
            self.ok = False


@dataclass
class CompareReport:
    info_notes: list[str] = field(default_factory=list)
    inputs: SectionReport = field(default_factory=SectionReport)
    outputs: SectionReport = field(default_factory=SectionReport)

    @property
    def ok(self) -> bool:
        return self.inputs.ok and self.outputs.ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MTP dumps: pre-forward inputs (strict), then outputs (cosine).",
    )
    parser.add_argument("ref_pt", type=Path, help="Reference (golden) .pt file.")
    parser.add_argument("cmp_pt", type=Path, help="Compare (suspect) .pt file.")
    parser.add_argument(
        "--cosine-min",
        type=float,
        default=1.0,
        help="Min global cosine for hidden tensors (default: 1.0).",
    )
    parser.add_argument(
        "--embeds-atol",
        type=float,
        default=0.0,
        help="atol for inputs_embeds float compare (default: 0).",
    )
    parser.add_argument(
        "--print-contents",
        action="store_true",
        help="Print tensor contents before compare.",
    )
    parser.add_argument(
        "--full-hidden",
        action="store_true",
        help="With --print-contents, print full hidden_states.",
    )
    parser.add_argument(
        "--max-tensor-elems",
        type=int,
        default=128,
        help="Max elems when printing tensors (default: 128).",
    )
    return parser.parse_args()


def load_record(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_tensor(tensor: torch.Tensor | None, *, max_elems: int, full: bool = False) -> str:
    if tensor is None:
        return "None"
    t = tensor.detach().cpu()
    header = f"dtype={t.dtype} shape={tuple(t.shape)}"
    if t.numel() == 0:
        return f"{header} []"
    if full or t.numel() <= max_elems:
        return f"{header} {t.flatten().tolist()}"
    flat = t.flatten()
    suffix = f" ... (+{t.numel() - max_elems})" if t.numel() > max_elems else ""
    return f"{header} head={flat[:max_elems].tolist()}{suffix}"


def _dtype_note(ref: torch.Tensor, cmp: torch.Tensor) -> str:
    if ref.dtype == cmp.dtype:
        return ""
    return f" [dtype ref={ref.dtype} cmp={cmp.dtype}, values compared]"


def _first_value_diff(ref: torch.Tensor, cmp: torch.Tensor) -> str | None:
    r = ref.detach().cpu().flatten()
    c = cmp.detach().cpu().flatten()
    if ref.is_floating_point():
        diff = (r.float() - c.float()).abs()
        if diff.max() == 0:
            return None
        idx = int(diff.argmax())
        return f"first_diff_idx={idx} ref={r[idx].item()} cmp={c[idx].item()} max_abs={diff.max().item():.6g}"
    r64 = r.to(torch.int64)
    c64 = c.to(torch.int64)
    mask = r64 != c64
    if not mask.any():
        return None
    idx = int(mask.argmax())
    return f"first_diff_idx={idx} ref={r64[idx].item()} cmp={c64[idx].item()}"


def tensors_value_equal(
    ref: torch.Tensor,
    cmp: torch.Tensor,
    *,
    embeds_atol: float = 0.0,
) -> tuple[bool, str]:
    ref = ref.detach().cpu()
    cmp = cmp.detach().cpu()
    if ref.shape != cmp.shape:
        return False, f"shape ref={tuple(ref.shape)} cmp={tuple(cmp.shape)}"

    note = _dtype_note(ref, cmp)
    if ref.is_floating_point() or cmp.is_floating_point():
        rf, cf = ref.float(), cmp.float()
        if embeds_atol > 0:
            if torch.allclose(rf, cf, rtol=0.0, atol=embeds_atol):
                return True, f"OK (allclose atol={embeds_atol}){note}"
        if torch.equal(rf, cf):
            return True, f"OK{note}"
        hint = _first_value_diff(ref, cmp) or ""
        return False, f"values differ{note}" + (f"; {hint}" if hint else "")

    if torch.equal(ref.to(torch.int64), cmp.to(torch.int64)):
        return True, f"OK{note}"
    hint = _first_value_diff(ref, cmp) or ""
    return False, f"values differ{note}" + (f"; {hint}" if hint else "")


def compare_optional_tensor(
    section: SectionReport,
    name: str,
    ref: torch.Tensor | None,
    cmp: torch.Tensor | None,
    *,
    embeds_atol: float,
) -> None:
    if ref is None and cmp is None:
        section.add(name, True, "OK (both None)")
        return
    if ref is None or cmp is None:
        section.add(name, False, f"ref_is_none={ref is None} cmp_is_none={cmp is None}")
        return
    ok, detail = tensors_value_equal(ref, cmp, embeds_atol=embeds_atol)
    section.add(name, ok, detail)


def compare_inputs(
    ref_rec: dict[str, Any],
    cmp_rec: dict[str, Any],
    *,
    embeds_atol: float,
) -> SectionReport:
    section = SectionReport()
    ref_in = ref_rec.get("inputs") or {}
    cmp_in = cmp_rec.get("inputs") or {}

    for key in FORWARD_INPUT_KEYS:
        name = f"inputs.{key}"
        ref_t, cmp_t = ref_in.get(key), cmp_in.get(key)
        if ref_t is None and cmp_t is None:
            section.add(name, True, "OK (both None)")
            continue
        if ref_t is None or cmp_t is None:
            section.add(name, False, f"ref_is_none={ref_t is None} cmp_is_none={cmp_t is None}")
            continue
        ok, detail = tensors_value_equal(ref_t, cmp_t)
        section.add(name, ok, detail)

    for key in FORWARD_INPUT_OPTIONAL_KEYS:
        compare_optional_tensor(
            section,
            f"inputs.{key}",
            ref_in.get(key),
            cmp_in.get(key),
            embeds_atol=embeds_atol,
        )

    ref_spec = ref_in.get("spec_decode")
    cmp_spec = cmp_in.get("spec_decode")
    if ref_spec is None and cmp_spec is None:
        section.add("inputs.spec_decode", True, "OK (both None)")
    elif ref_spec is None or cmp_spec is None:
        section.add(
            "inputs.spec_decode",
            False,
            f"ref_is_none={ref_spec is None} cmp_is_none={cmp_spec is None}",
        )
    else:
        for key in SPEC_INPUT_KEYS:
            name = f"inputs.spec_decode.{key}"
            ref_t, cmp_t = ref_spec.get(key), cmp_spec.get(key)
            if ref_t is None and cmp_t is None:
                section.add(name, True, "OK (both None)")
                continue
            if ref_t is None or cmp_t is None:
                section.add(name, False, f"ref_is_none={ref_t is None} cmp_is_none={cmp_t is None}")
                continue
            ok, detail = tensors_value_equal(ref_t, cmp_t)
            section.add(name, ok, detail)

    return section


def cosine_global(ref: torch.Tensor, cmp: torch.Tensor) -> float:
    ref_f = ref.detach().float().cpu().flatten()
    cmp_f = cmp.detach().float().cpu().flatten()
    if ref_f.numel() == 0:
        return 1.0
    return float(F.cosine_similarity(ref_f.unsqueeze(0), cmp_f.unsqueeze(0)).item())


def cosine_per_token_stats(ref: torch.Tensor, cmp: torch.Tensor) -> tuple[float, float] | None:
    ref_f = ref.detach().float().cpu()
    cmp_f = cmp.detach().float().cpu()
    if ref_f.dim() < 2 or ref_f.shape[0] == 0:
        return None
    ref_rows = ref_f.reshape(ref_f.shape[0], -1)
    cmp_rows = cmp_f.reshape(cmp_f.shape[0], -1)
    if ref_rows.shape != cmp_rows.shape:
        return None
    per = F.cosine_similarity(ref_rows, cmp_rows, dim=-1)
    return float(per.min().item()), float(per.mean().item())


def compare_outputs(
    ref_rec: dict[str, Any],
    cmp_rec: dict[str, Any],
    *,
    cosine_min: float,
) -> SectionReport:
    section = SectionReport()
    ref_out = ref_rec.get("outputs") or {}
    cmp_out = cmp_rec.get("outputs") or {}

    for key in OUTPUT_COSINE_KEYS:
        name = f"outputs.{key}"
        ref_t, cmp_t = ref_out.get(key), cmp_out.get(key)
        if ref_t is None and cmp_t is None:
            section.add(name, True, "OK (both None)")
            continue
        if ref_t is None or cmp_t is None:
            section.add(name, False, f"ref_is_none={ref_t is None} cmp_is_none={cmp_t is None}")
            continue
        if ref_t.shape != cmp_t.shape:
            section.add(
                name,
                False,
                f"shape ref={tuple(ref_t.shape)} cmp={tuple(cmp_t.shape)}",
            )
            continue

        g = cosine_global(ref_t, cmp_t)
        parts = [f"global_cosine={g:.8f} (need>={cosine_min})"]
        pt = cosine_per_token_stats(ref_t, cmp_t)
        if pt is not None:
            parts.append(f"per_token_min={pt[0]:.8f} per_token_mean={pt[1]:.8f} [diagnostic]")
        ok = g >= cosine_min
        section.add(name, ok, "OK " + ", ".join(parts) if ok else "FAIL " + ", ".join(parts))

    return section


def collect_info_notes(ref_rec: dict[str, Any], cmp_rec: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for key in INFO_KEYS:
        ref_v = ref_rec.get(key) if key != "req_ids" else ref_rec.get("req_ids")
        cmp_v = cmp_rec.get(key) if key != "req_ids" else cmp_rec.get("req_ids")
        if ref_v != cmp_v:
            notes.append(f"{key}: ref={ref_v!r} cmp={cmp_v!r}")
    return notes


def print_section(title: str, section: SectionReport) -> None:
    print(title)
    status = "PASS" if section.ok else "FAIL"
    print(f"  >> {status}")
    for f in section.fields:
        mark = "OK" if f.ok else "FAIL"
        print(f"  [{mark}] {f.name}: {f.detail}")
    print()


def print_final(report: CompareReport) -> None:
    if report.ok:
        print("RESULT: PASS — pre-forward inputs match; outputs meet cosine threshold.")
        return
    if not report.inputs.ok:
        print("RESULT: FAIL — pre-forward inputs differ (see [FAIL] lines above).")
        if report.outputs.ok:
            print("         outputs OK; fix inputs before trusting output diff.")
        else:
            print("         outputs also differ; fix inputs first.")
        return
    print("RESULT: FAIL — pre-forward inputs match, but hidden states diverge.")
    print("         suspect forward ops / KV / GDN (not scheduler input layout).")


def compare_record(
    ref_rec: dict[str, Any],
    cmp_rec: dict[str, Any],
    *,
    cosine_min: float,
    embeds_atol: float,
) -> CompareReport:
    report = CompareReport()
    report.info_notes = collect_info_notes(ref_rec, cmp_rec)
    report.inputs = compare_inputs(ref_rec, cmp_rec, embeds_atol=embeds_atol)
    report.outputs = compare_outputs(ref_rec, cmp_rec, cosine_min=cosine_min)
    return report


def main() -> int:
    args = parse_args()
    ref_path = args.ref_pt.expanduser().resolve()
    cmp_path = args.cmp_pt.expanduser().resolve()

    if not ref_path.is_file():
        print(f"ERROR: ref not found: {ref_path}", file=sys.stderr)
        return 2
    if not cmp_path.is_file():
        print(f"ERROR: cmp not found: {cmp_path}", file=sys.stderr)
        return 2

    ref_rec = load_record(ref_path)
    cmp_rec = load_record(cmp_path)

    print(f"REF: {ref_path}")
    print(f"CMP: {cmp_path}")
    print(f"cosine_min={args.cosine_min}")
    print()

    if args.print_contents:
        for label, rec in ("REF", ref_rec), ("CMP", cmp_rec):
            print(f"--- {label} ---")
            for k in INFO_KEYS:
                print(f"  {k}={rec.get(k)}")
            inputs = rec.get("inputs") or {}
            for key in FORWARD_INPUT_KEYS + FORWARD_INPUT_OPTIONAL_KEYS:
                print(f"  inputs.{key}: {format_tensor(inputs.get(key), max_elems=args.max_tensor_elems)}")
            spec = inputs.get("spec_decode")
            if spec:
                for key in SPEC_INPUT_KEYS:
                    print(f"  inputs.spec_decode.{key}: {format_tensor(spec.get(key), max_elems=args.max_tensor_elems)}")
            outputs = rec.get("outputs") or {}
            for key in OUTPUT_COSINE_KEYS:
                full = args.full_hidden and key == "hidden_states"
                print(f"  outputs.{key}: {format_tensor(outputs.get(key), max_elems=args.max_tensor_elems, full=full)}")
            print()

    report = compare_record(ref_rec, cmp_rec, cosine_min=args.cosine_min, embeds_atol=args.embeds_atol)

    if report.info_notes:
        print("=== Context (not scored: step / path / req_ids / num_tokens) ===")
        for line in report.info_notes:
            print(f"  {line}")
        print()

    print_section("=== [1/2] Pre-forward INPUT (strict, value-equal) ===", report.inputs)
    print_section("=== [2/2] Forward OUTPUT (global cosine) ===", report.outputs)
    print_final(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

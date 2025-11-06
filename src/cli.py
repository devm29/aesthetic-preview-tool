"""Headless entry point: run the whole pipeline without Streamlit.

    python -m src.cli assets/sample_face.jpg --wrinkle 60 --out out.png

Useful for batch checks and for proving that the image pipeline does not depend
on the UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import DISCLAIMER, get_settings
from .generative import build_editor
from .imaging import encode_png
from .pipeline import (
    analyse,
    build_export,
    describe_result,
    load_preview_image,
    render_preview,
)
from .treatments import all_treatments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Render an aesthetic-preview edit headlessly. " + DISCLAIMER,
    )
    parser.add_argument("image", type=Path, help="Path to a JPEG or PNG portrait.")
    parser.add_argument("--out", type=Path, help="Write a before/after PNG here.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.add_argument("--force", action="store_true", help="Render even if quality checks fail.")
    # One flag per registered treatment - new treatments appear automatically.
    for treatment in all_treatments():
        parser.add_argument(
            f"--{treatment.key.replace('_', '-')}",
            type=int,
            default=None,
            metavar="0-100",
            help=f"{treatment.spec.label}: {treatment.spec.summary}",
        )
    return parser


def _intensities(args: argparse.Namespace) -> dict[str, int]:
    values: dict[str, int] = {}
    for treatment in all_treatments():
        given = getattr(args, treatment.key, None)
        values[treatment.key] = treatment.spec.default if given is None else int(given)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    if not args.image.is_file():
        print(f"No such image: {args.image}", file=sys.stderr)
        return 2

    image = load_preview_image(args.image.read_bytes(), settings)
    report = analyse(image, settings)

    lines: list[str] = [f"Image      : {args.image} ({image.shape[1]}x{image.shape[0]})"]
    for check in report.checks:
        marker = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[check.status]
        value = f" ({check.value:.1f} {check.unit})" if check.value is not None else ""
        lines.append(f"  [{marker}] {check.name}{value}: {check.detail}")

    if report.face is None:
        print("\n".join(lines))
        print("\nNo face detected; nothing to render.")
        return 1
    if report.failures and not args.force:
        print("\n".join(lines))
        print("\nQuality checks failed. Re-run with --force to render anyway.")
        return 1

    result = render_preview(image, report, _intensities(args), settings, build_editor(settings))
    summary = describe_result(result, report)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        lines.append(f"Detector   : {report.face.detector}")
        for applied in result.applied:
            lines.append(
                f"  applied  {applied.label} @ {applied.intensity}% "
                f"({applied.engine}, {applied.elapsed_ms:.0f} ms)"
            )
        if not result.applied:
            lines.append("  applied  nothing (all intensities were zero)")
        for note in result.notes:
            lines.append(f"  note     {note}")
        if result.identity:
            lines.append(
                f"Identity   : SSIM {result.identity.ssim:.3f} "
                f"(threshold {result.identity.threshold:.2f}) - {result.identity.verdict}"
            )
        lines.append(f"Total      : {result.elapsed_ms:.0f} ms")
        lines.append(f"\n{DISCLAIMER}")
        print("\n".join(lines))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(encode_png(build_export(result, settings)))
        if not args.json:
            print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

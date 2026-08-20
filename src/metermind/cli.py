from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("matplotlib", "numexpr", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _banner(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def _timed(label: str, fn: Callable, *args, **kwargs):
    _banner(label)
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"\n  -> {label} finished in {time.perf_counter() - started:,.1f}s")
    return result


def cmd_ingest(args: argparse.Namespace) -> int:
    from metermind.ingest import ingest_lcl

    _timed("EXTRACT  reduced Low Carbon London export into bronze", ingest_lcl)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from metermind.warehouse import build_warehouse

    _timed("TRANSFORM  bronze to silver to gold", build_warehouse, rebuild=args.rebuild)
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    from metermind.quality import run_quality_suite

    report = _timed("VALIDATE  data quality suite", run_quality_suite)
    return 0 if report.passed else 2


def cmd_train(args: argparse.Namespace) -> int:
    from metermind.pipeline import train_all

    _timed(
        "TRAIN  bill forecaster, shape autoencoder, causal uplift",
        train_all, quick=args.quick, limit_households=args.limit,
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from metermind.warehouse.export import export_for_app

    _timed("EXPORT  slim app database", export_for_app)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_ingest, cmd_build, cmd_quality, cmd_train, cmd_export):
        code = step(args)
        if code != 0 and step is not cmd_quality:
            return code
    _banner("PIPELINE COMPLETE")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metermind",
        description="Residential smart meter intelligence pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: ingest, build, quality, train, export, all.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="load the reduced export into bronze").set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build", help="build the silver and gold layers")
    p_build.add_argument("--rebuild", action="store_true", help="drop and recreate gold tables")
    p_build.set_defaults(func=cmd_build)

    sub.add_parser("quality", help="run the data quality suite").set_defaults(func=cmd_quality)

    p_train = sub.add_parser("train", help="train all three models and write artifacts")
    p_train.add_argument("--quick", action="store_true", help="fewer epochs and trees, for CI")
    p_train.add_argument("--limit", type=int, default=None, help="cap the number of households")
    p_train.set_defaults(func=cmd_train)

    sub.add_parser("export", help="write the slim database the app ships with").set_defaults(
        func=cmd_export
    )

    p_all = sub.add_parser("all", help="run every stage end to end")
    p_all.add_argument("--rebuild", action="store_true")
    p_all.add_argument("--quick", action="store_true")
    p_all.add_argument("--limit", type=int, default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    for flag in ("rebuild", "quick"):
        setattr(args, flag, getattr(args, flag, False))
    args.limit = getattr(args, "limit", None)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logging.getLogger("metermind").error(
            "%s: %s", type(exc).__name__, exc, exc_info=args.verbose
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

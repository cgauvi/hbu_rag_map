"""``python -m eval`` - the harness's command line.

    python -m eval cases [--write]              the question set (regenerate it from the database)
    python -m eval run --config NAME            one configuration, in this process
    python -m eval compare [--configs a,b,c]    every configuration, each in its own process, then the report
    python -m eval retrieval --embedder bge|voyage   the encoder alone
    python -m eval reembed [--model voyage-3.5]      the corpus through the closed encoder (resumable)
    python -m eval shadow build|drop|status          the rag_eval schema
    python -m eval report                             rebuild report.md from what is on disk
    python -m eval regrade                            recompute the free checks on stored rows
    python -m eval spend                              what has been spent

Everything lands under ``eval_out/`` (``--out`` to change it), which is
gitignored. The Anthropic cap is ``--budget-usd`` (default $5.00), cumulative
over every run in that directory, and it stops a run before the call that
would pass it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUT = REPO / "eval_out"


def _load_env() -> None:
    """``.env`` without overriding what the shell - or a configuration - set."""
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(REPO / ".env", override=False)
    # The proxy on this laptop: the HF client reads SSL_CERT_FILE, and `.env`
    # names a bundle that exists on Windows. Nothing to do when it is set.


def _connect():
    import psycopg  # noqa: PLC0415

    dsn = os.environ.get("DATABASE_URL") or os.environ.get("URBAN_RAG_PG_DSN")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set (the local container: `make db-up` in hbu_rag_map)")
    return psycopg.connect(dsn, connect_timeout=15)


def _meter(args):
    from eval.spend import SpendMeter  # noqa: PLC0415

    return SpendMeter.open(Path(args.out), cap_usd=args.budget_usd)


def _cases(args):
    from eval import golden  # noqa: PLC0415

    cases = golden.load()
    only = [s.strip() for s in (args.only or "").split(",") if s.strip()] or None
    categories = [s.strip() for s in (args.categories or "").split(",") if s.strip()] or None
    return golden.select(cases, only=only, categories=categories, limit=args.limit)


# -- subcommands ---------------------------------------------------------------


def cmd_cases(args) -> int:
    from eval import golden  # noqa: PLC0415

    if args.write:
        _load_env()
        with _connect() as conn:
            cases = golden.generate(conn)
        golden.dump(cases)
        print(f"wrote {len(cases)} cases to {golden.FIXTURE}")
    cases = golden.load()
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    print(f"{len(cases)} cases: " + ", ".join(f"{k} {v}" for k, v in by_cat.items()))
    if args.verbose:
        for c in cases:
            lot = f" @ {c.lot['lot_number']} ({c.lot['neighborhood']})" if c.lot else ""
            print(f"  {c.id:<40} {c.category:<16}{lot}\n      {c.question}\n      expect: {json.dumps(c.expect, ensure_ascii=False)}")
    return 0


def cmd_run(args) -> int:
    from eval import configs  # noqa: PLC0415

    config = configs.get(args.config)
    configs.apply(config)
    _load_env()
    import logging  # noqa: PLC0415

    logging.basicConfig(level=logging.WARNING)
    from eval.runner import Harness  # noqa: PLC0415

    meter = _meter(args)
    cases = _cases(args)
    out_dir = Path(args.out) / config.name
    harness = Harness(config, meter, out_dir=out_dir, judge=not args.no_judge, judge_model=args.judge_model, timeout_s=args.timeout_s)
    summary = harness.run(cases, reps=args.reps, retries=args.retries)
    print(json.dumps(summary, indent=1))
    from eval import report  # noqa: PLC0415

    print(f"report: {report.write(Path(args.out))}")
    return 2 if summary.get("stopped") else 0


def cmd_compare(args) -> int:
    names = [s.strip() for s in args.configs.split(",") if s.strip()]
    code = 0
    for name in names:
        cmd = [sys.executable, "-X", "utf8", "-m", "eval", "run", "--config", name, "--out", args.out, "--budget-usd", str(args.budget_usd),
               "--reps", str(args.reps), "--timeout-s", str(args.timeout_s), "--judge-model", args.judge_model, "--retries", str(args.retries)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.only:
            cmd += ["--only", args.only]
        if args.categories:
            cmd += ["--categories", args.categories]
        if args.no_judge:
            cmd.append("--no-judge")
        print("$ " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO, check=False)
        code = max(code, result.returncode)
    from eval import report  # noqa: PLC0415

    print(f"report: {report.write(Path(args.out))}")
    return code


def cmd_retrieval(args) -> int:
    from eval import configs  # noqa: PLC0415

    config = configs.EMBEDDERS[args.embedder]
    configs.apply(config)
    _load_env()
    from eval import retrieval  # noqa: PLC0415

    meter = _meter(args)
    cases = _cases(args)
    summary = retrieval.run(config, cases, out_dir=Path(args.out) / "retrieval", meter=meter)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_reembed(args) -> int:
    _load_env()
    from eval import corpus  # noqa: PLC0415

    meter = _meter(args)
    neighborhoods = [s.strip() for s in (args.neighborhoods or "").split(",") if s.strip()] or None
    with _connect() as conn:
        summary = corpus.reembed(conn, model=args.model, neighborhoods=neighborhoods, batch=args.batch, meter=meter,
                                 limit=args.limit, free_tier=args.free_tier)
    print(json.dumps(summary, indent=1))
    print(json.dumps(meter.summary(), indent=1))
    return 0


def cmd_shadow(args) -> int:
    _load_env()
    from eval import shadow  # noqa: PLC0415

    with _connect() as conn:
        if args.action == "build":
            print(json.dumps(shadow.build(conn, args.model, allow_partial=args.allow_partial), indent=1))
        elif args.action == "drop":
            shadow.drop(conn)
            print("dropped rag_eval")
        else:
            print(json.dumps(shadow.status(conn), indent=1))
    return 0


def cmd_report(args) -> int:
    from eval import report  # noqa: PLC0415

    path = report.write(Path(args.out))
    print(path.read_text(encoding="utf-8"))
    print(f"\nwritten to {path}")
    return 0


def cmd_regrade(args) -> int:
    _load_env()
    from eval import report  # noqa: PLC0415
    from eval.runner import regrade  # noqa: PLC0415

    out = Path(args.out)
    meter = _meter(args) if args.rejudge else None
    total = 0
    for path in sorted(out.glob("*/results.jsonl")):
        if args.configs and path.parent.name not in args.configs.split(","):
            continue
        n = regrade(path, rejudge=args.rejudge, meter=meter, judge_model=args.judge_model)
        total += n
        print(f"regraded {n} row(s) in {path}")
    print(f"report: {report.write(out)}")
    if meter is not None:
        print(json.dumps(meter.summary(), indent=1))
    return 0 if total else 1


def cmd_spend(args) -> int:
    meter = _meter(args)
    print(json.dumps(meter.summary(), indent=1))
    by_model: dict[str, dict] = {}
    for entry in meter.entries:
        slot = by_model.setdefault(entry["model"], {"calls": 0, "usd": 0.0, "input": 0, "output": 0})
        slot["calls"] += 1
        slot["usd"] += entry["usd"]
        slot["input"] += entry["usage"].get("input_tokens", 0) + entry["usage"].get("cache_read_input_tokens", 0)
        slot["output"] += entry["usage"].get("output_tokens", 0)
    for model, slot in sorted(by_model.items()):
        print(f"  {model:<32} {slot['calls']:>5} calls  {slot['input']:>10,} in  {slot['output']:>8,} out  ${slot['usd']:.4f}")
    return 0


# -- parser -------------------------------------------------------------------


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default=str(DEFAULT_OUT), help="results directory (default eval_out/)")
    p.add_argument("--budget-usd", type=float, default=float(os.environ.get("HBU_EVAL_BUDGET_USD", 5.0)),
                   help="cap on cumulative Anthropic spend in --out (default 5.00)")


def _selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--limit", type=int, default=None, help="first N cases only")
    p.add_argument("--only", default=None, help="comma-separated case ids")
    p.add_argument("--categories", default=None, help="comma-separated categories")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cases", help="list, or regenerate, the question set")
    p.add_argument("--write", action="store_true", help="regenerate tests/fixtures/agent_eval.yaml from the database")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("run", help="one configuration through the agent")
    p.add_argument("--config", required=True)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default=os.environ.get("HBU_EVAL_JUDGE_MODEL", "claude-haiku-5-5"))
    p.add_argument("--timeout-s", type=float, default=300.0, help="hard wall-clock ceiling per case")
    p.add_argument("--retries", type=int, default=2, help="re-runs of a case after a serving error (429, dropped connection)")
    _selection(p)
    _common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="every configuration, then the report")
    p.add_argument("--configs", default="open,claude-bge,claude-voyage")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default=os.environ.get("HBU_EVAL_JUDGE_MODEL", "claude-haiku-5-5"))
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=2)
    _selection(p)
    _common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("retrieval", help="the encoder alone, no chat model")
    p.add_argument("--embedder", choices=("bge", "voyage"), required=True)
    _selection(p)
    _common(p)
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("reembed", help="embed the corpus with the closed encoder (resumable)")
    p.add_argument("--model", default="voyage-3.5")
    p.add_argument("--neighborhoods", default=None, help="comma-separated borough codes; default all")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--free-tier", action="store_true",
                   help="pace under Voyage's no-payment-method limits (3 RPM, 10K TPM): ~9K tokens a minute")
    _common(p)
    p.set_defaults(func=cmd_reembed)

    p = sub.add_parser("shadow", help="the rag_eval schema")
    p.add_argument("action", choices=("build", "drop", "status"))
    p.add_argument("--model", default="voyage-3.5")
    p.add_argument("--allow-partial", action="store_true")
    _common(p)
    p.set_defaults(func=cmd_shadow)

    p = sub.add_parser("report", help="rebuild report.md")
    _common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("regrade", help="recompute the free checks on every stored row (judge columns kept unless --rejudge)")
    p.add_argument("--rejudge", action="store_true", help="run the judge again on the stored answers (costs Anthropic spend)")
    p.add_argument("--configs", default=None, help="comma-separated configuration names; default all on disk")
    p.add_argument("--judge-model", default=os.environ.get("HBU_EVAL_JUDGE_MODEL", "claude-haiku-5-5"))
    _common(p)
    p.set_defaults(func=cmd_regrade)

    p = sub.add_parser("spend", help="what has been spent")
    _common(p)
    p.set_defaults(func=cmd_spend)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

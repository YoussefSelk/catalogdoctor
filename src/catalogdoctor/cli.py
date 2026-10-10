"""Dashboard launcher and scriptable report generation."""

import argparse
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from catalogdoctor.demo import default_rules, generate_catalog
from catalogdoctor.ingestion import load_file, load_sqlite
from catalogdoctor.reporting import analyze, export_json


def main() -> None:
    parser = argparse.ArgumentParser(prog="catalogdoctor")
    commands = parser.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser("dashboard", help="Launch Streamlit")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--address", default="localhost")
    scan = commands.add_parser("scan", help="Write a JSON analysis report")
    scan.add_argument("input", nargs="?", help="CSV, Parquet or SQLite file; omitted uses demo")
    scan.add_argument("--table", help="Reflected SQLite table name")
    scan.add_argument("--rules", type=Path)
    scan.add_argument("--output", type=Path, default=Path("catalogdoctor-report.json"))
    args = parser.parse_args()
    if args.command == "dashboard":
        app = files("catalogdoctor.dashboard").joinpath("app.py")
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app),
                    "--server.port",
                    str(args.port),
                    "--server.address",
                    args.address,
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ]
            )
        )
    try:
        df = (
            (load_sqlite(args.input, args.table)[0] if args.table else load_file(args.input))
            if args.input
            else generate_catalog()[0]
        )
        rules = (
            args.rules.read_text(encoding="utf-8")
            if args.rules
            else (default_rules() if args.input is None else "rules: []")
        )
        args.output.write_text(export_json(analyze(df, rules)), encoding="utf-8")
        print(f"Wrote {args.output}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

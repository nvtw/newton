# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run selected regression modules in isolated processes and retain every result."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Keep module logs, unittest summaries, exit codes, and source integrity."""
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def fingerprints():
        paths = list((root / "newton").rglob("*.py"))
        paths += [p for p in (root / "newton/examples/assets/colibri").rglob("*") if p.is_file()]
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

    original = fingerprints()
    report = {"source_asset_sha256": original, "requested": args.modules, "cases": [], "passed": False}
    try:
        for index, module in enumerate(args.modules):
            log = args.output.with_name(f"{args.output.stem}_{index:02d}_{module.rsplit('.', 1)[-1]}.log")
            command = ["uv", "run", "--no-project", sys.executable, "-m", "unittest", "-v", module]
            print("START", module, flush=True)
            start = time.perf_counter()
            with log.open("w") as stream:
                process = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
            lines = log.read_text(errors="replace").splitlines()
            summaries = [line for line in lines if re.match(r"^(Ran \d+ tests? in |OK(?:$| \()|FAILED \()", line)]
            count_match = re.search(r"Ran (\d+) tests? in ", "\n".join(summaries))
            test_count = int(count_match.group(1)) if count_match else 0
            unchanged = fingerprints() == original
            case = {
                "module": module,
                "command": command,
                "log": str(log),
                "returncode": process.returncode,
                "test_count": test_count,
                "passed": process.returncode == 0 and test_count > 0,
                "elapsed_seconds": time.perf_counter() - start,
                "unittest_summary": summaries,
                "source_assets_unchanged": unchanged,
            }
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2))
            print("PASS" if case["passed"] else "FAIL", module, summaries, flush=True)
            if not unchanged:
                raise RuntimeError("Production source or assets changed during regression run")
            if not case["passed"] and not args.keep_going:
                break
        report["passed"] = len(report["cases"]) == len(args.modules) and all(case["passed"] for case in report["cases"])
    finally:
        report["source_assets_unchanged"] = fingerprints() == original
        args.output.write_text(json.dumps(report, indent=2))
    if not report["passed"] or not report["source_assets_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

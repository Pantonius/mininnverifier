# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Test runner for mininnverifier-compatible implementations.

Usage:
    python run_test.py <test_dir> docker <image>
    python run_test.py <test_dir> local "<command>"
    python run_test.py <test_dir> docker <image> --generate
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------
# Each builder returns (cmd_list, cwd_or_None).
# For docker: cmd includes docker run with volume mount, cwd is None.
# For local: cmd uses relative paths, cwd is the test dir.


def _build_eval_grad_cmd(config, test_dir, output_dir, backend, backend_arg):
    """Shared builder for eval and grad — same argument structure."""
    command = config["command"]
    network = config["network"]
    inputs = config.get("inputs", [])

    if backend == "docker":
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{test_dir.resolve()}:/data",
            backend_arg,
            command,
            "--output-dir", f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network}",
            *[f"/data/{i}" for i in inputs],
        ]
        return cmd, None
    else:
        cmd = [
            *shlex.split(backend_arg),
            command,
            "--output-dir", str(output_dir),
            *[str(test_dir / i) for i in [network, *inputs]],
        ]
        return cmd, None


def build_eval_cmd(config, test_dir, output_dir, backend, backend_arg):
    return _build_eval_grad_cmd(config, test_dir, output_dir, backend, backend_arg)


def build_grad_cmd(config, test_dir, output_dir, backend, backend_arg):
    return _build_eval_grad_cmd(config, test_dir, output_dir, backend, backend_arg)


COMMANDS = {
    "eval": build_eval_cmd,
    "grad": build_grad_cmd,
}

# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------
# Each check returns {"passed": bool, "error": str | None}.


def check_close_to_expected(test_dir, config, output_files):
    expected_names = config.get("expected_outputs", [])
    tolerance = config.get("tolerance", 1e-9)

    if len(output_files) != len(expected_names):
        return {
            "passed": False,
            "error": (
                f"expected {len(expected_names)} output file(s), "
                f"got {len(output_files)}"
            ),
        }

    for out_file, exp_name in zip(output_files, expected_names):
        actual = np.fromfile(out_file, dtype=np.float64)
        expected = np.fromfile(test_dir / exp_name, dtype=np.float64)

        if actual.shape != expected.shape:
            return {
                "passed": False,
                "error": (
                    f"{out_file.name}: shape mismatch: "
                    f"actual {actual.shape} vs expected {expected.shape}"
                ),
            }

        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            diff = np.abs(actual - expected)
            worst = int(np.argmax(diff))
            return {
                "passed": False,
                "error": (
                    f"{out_file.name}: max absolute diff "
                    f"{diff[worst]:.6e} > tolerance {tolerance:.1e} "
                    f"(index {worst}: expected {expected[worst]:.6e}, "
                    f"got {actual[worst]:.6e})"
                ),
            }

    return {"passed": True, "error": None}


CHECKS = {
    "close_to_expected": check_close_to_expected,
}

DEFAULT_CHECKS = {
    "eval": "close_to_expected",
    "grad": "close_to_expected",
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(test_dir, backend, backend_arg, generate=False):
    test_dir = Path(test_dir).resolve()
    config = json.loads((test_dir / "test.json").read_text())

    command = config["command"]
    if command not in COMMANDS:
        return {"test": test_dir.name, "passed": False,
                "error": f"unknown command: {command}"}

    output_dir = test_dir / "actual"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    cmd, cwd = COMMANDS[command](config, test_dir, output_dir, backend, backend_arg)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0:
        return {
            "test": test_dir.name,
            "passed": False,
            "output_files": [],
            "error": f"command failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    output_files = [
        Path(line) for line in result.stdout.strip().splitlines() if line.strip()
    ]

    if generate:
        for out_file in output_files:
            dest = test_dir / f"expected_{out_file.name}"
            shutil.copy2(out_file, dest)
        return {
            "test": test_dir.name,
            "generated": True,
            "output_files": [str(f) for f in output_files],
        }

    check_name = config.get("check", DEFAULT_CHECKS.get(command))
    if check_name is None:
        return {"test": test_dir.name, "passed": False,
                "error": f"no check specified and no default for command '{command}'"}
    if check_name not in CHECKS:
        return {"test": test_dir.name, "passed": False,
                "error": f"unknown check: {check_name}"}

    check_result = CHECKS[check_name](test_dir, config, output_files)
    return {
        "test": test_dir.name,
        "passed": check_result["passed"],
        "output_files": [str(f) for f in output_files],
        "error": check_result["error"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run a mininnverifier test.")
    parser.add_argument("test_dir", type=str)
    parser.add_argument("backend", choices=["docker", "local"])
    parser.add_argument("backend_arg", type=str,
                        help="Docker image name or local command")
    parser.add_argument("--generate", action="store_true",
                        help="Generate expected outputs instead of checking")
    args = parser.parse_args()

    result = run_test(args.test_dir, args.backend, args.backend_arg, args.generate)
    print(json.dumps(result, indent=2))
    if not result.get("passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()

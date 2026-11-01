#!/usr/bin/env python3
"""Install a built wheel in a fresh venv and exercise it from outside the source checkout.

Checks: license files in the wheel, CLI help for every subcommand, workspace resolution
(default and ``BREAKPOINT_ROOT``) with a read-only installed package, and optionally rendering
the saved example with the installed code.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

LOGGER = logging.getLogger("check_wheel")

ROOT_SNIPPET = (
    "import breakpoint_eval.stack as s; print(s.__file__); print(s.repo_root_from_package())"
)


def run(cmd: list[str | Path], *, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def check_license_files(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    for required in ("LICENSE", "NOTICE"):
        if not any(n.endswith(f".dist-info/licenses/{required}") for n in names):
            raise SystemExit(f"{required} missing from {wheel.name}")
    LOGGER.info("LICENSE and NOTICE are in the wheel")


def set_writable(path: Path, *, writable: bool) -> None:
    mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for item in [path, *path.rglob("*")]:
        mode = item.stat().st_mode
        item.chmod(mode | stat.S_IWUSR if writable else mode & ~mask)


def check_installed(wheel: Path, example_data: list[Path]) -> None:
    with tempfile.TemporaryDirectory(prefix="breakpoint-wheel-") as tmp_name:
        tmp = Path(tmp_name)
        env_dir, outside, home, workspace = (tmp / n for n in ("venv", "outside", "home", "ws"))
        for d in (outside, home, workspace):
            d.mkdir()
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / "bin" / "python"
        dropped = ("BREAKPOINT_ROOT", "PYTHONPATH", "XDG_DATA_HOME")
        env = {k: v for k, v in os.environ.items() if k not in dropped}
        env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
        run([python, "-m", "pip", "install", "-q", str(wheel)], cwd=outside, env=env)

        package_dir = Path(
            run(
                [
                    python,
                    "-c",
                    "import breakpoint_eval, os; print(os.path.dirname(breakpoint_eval.__file__))",
                ],
                cwd=outside,
                env=env,
            ).strip()
        )
        if "site-packages" not in package_dir.parts:
            raise SystemExit(f"imported {package_dir}, expected an installed copy")
        set_writable(package_dir, writable=False)

        try:
            cli = python.parent / "breakpoint"
            choices = re.search(r"\{([a-z0-9,-]+)\}", run([cli, "--help"], cwd=outside, env=env))
            commands = choices.group(1).split(",") if choices else []
            if not commands:
                raise SystemExit("could not read subcommands from `breakpoint --help`")
            for command in commands:
                run([cli, command, "--help"], cwd=outside, env=env)
            LOGGER.info("core help ok: %s", ", ".join(commands))

            default_root = run([python, "-c", ROOT_SNIPPET], cwd=outside, env=env).splitlines()[1]
            expected = (home / ".local" / "share" / "breakpoint-eval").resolve()
            if Path(default_root).resolve() != expected:
                raise SystemExit(f"default workspace {default_root}, expected {expected}")
            override = run(
                [python, "-c", ROOT_SNIPPET],
                cwd=outside,
                env={**env, "BREAKPOINT_ROOT": str(workspace)},
            ).splitlines()[1]
            if Path(override).resolve() != workspace.resolve():
                raise SystemExit(f"BREAKPOINT_ROOT ignored: {override}")
            LOGGER.info("workspace: default %s, override %s", default_root, override)

            if example_data:
                set_writable(package_dir, writable=True)
                run(
                    [python, "-m", "pip", "install", "-q", f"{wheel}[report]"], cwd=outside, env=env
                )
                set_writable(package_dir, writable=False)
                script = Path(__file__).resolve().with_name("render_benchmark.py")
                for index, data_dir in enumerate(example_data):
                    report_dir = outside / f"report-{index}"
                    run(
                        [python, script, "--input", data_dir, "--out", report_dir],
                        cwd=outside,
                        env=env,
                    )
                    if not (report_dir / "index.html").is_file():
                        raise SystemExit(f"saved example did not render index.html: {data_dir}")
                    LOGGER.info("rendered saved example with the installed wheel: %s", data_dir)
            if any(workspace.iterdir()):
                raise SystemExit("commands wrote into the explicit workspace unexpectedly")

        finally:
            set_writable(package_dir, writable=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--example-data",
        type=Path,
        action="append",
        default=[],
        help="packaged benchmark data directory to render (repeatable)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    check_license_files(args.wheel)
    check_installed(
        args.wheel.resolve(),
        [path.resolve() for path in args.example_data],
    )
    LOGGER.info("wheel checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare upstream Python symbols against the Rust port.

This is a lightweight inventory checker for porting work. It scans top-level
Python functions plus direct class methods, scans Rust top-level functions plus
impl/trait methods, then reports Python symbols that do not have a Rust
counterpart.

The scanner is intentionally text-based instead of using ``ast``: upstream
Liquid Audio uses Python syntax newer than the macOS system Python parser.

Usage:
    python parity/compare_symbols.py
    python parity/compare_symbols.py --scope all
    python parity/compare_symbols.py --scope core --json /tmp/symbols.json
    python parity/compare_symbols.py --scope core --fail-on-missing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CRATE = HERE.parent
DEFAULT_PY = CRATE.parent / "upstream-liquid-audio" / "src" / "liquid_audio"
DEFAULT_RS = CRATE / "src"


PY_CLASS = re.compile(r"^(\s*)class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PY_DEF = re.compile(r"^(\s*)(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RS_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RS_TYPE = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RS_IMPL = re.compile(r"^\s*impl(?:<[^{}]*>)?\s+([^{}]+?)\s*\{")
RS_TRAIT = re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\b[^{}]*\{")
RS_CFG_TEST = re.compile(r"(?m)^\s*#\[cfg\(test\)\]\s*\n\s*mod\s+tests\s*\{")


@dataclass(frozen=True)
class Symbol:
    file: str
    line: int
    kind: str
    name: str
    canon: str
    cls: str | None = None

    @property
    def qual(self) -> str:
        if self.cls:
            return f"{self.cls}.{self.name}"
        return self.name


@dataclass
class RustFile:
    exists: bool
    top: set[str]
    methods: dict[str, set[str]]
    types: set[str]


def canon(name: str) -> str:
    aliases = {
        "__init__": "new",
        "__call__": "call",
        "__getitem__": "get",
        "__len__": "len",
        "__repr__": "fmt",
        "__iter__": "iter",
        "__next__": "next",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("_") and not name.startswith("__"):
        return name[1:]
    return name


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def strip_impl_type(text: str) -> str:
    value = text.strip()
    if " for " in value:
        value = value.split(" for ")[-1].strip()
    value = re.sub(r"<.*", "", value).strip()
    value = value.split()[0].split("::")[-1]
    return re.sub(r"[^A-Za-z0-9_].*", "", value)


def remove_rust_tests(text: str) -> str:
    match = RS_CFG_TEST.search(text)
    if match:
        return text[: match.start()]
    return text


def python_symbols(path: Path, root: Path) -> list[Symbol]:
    out: list[Symbol] = []
    classes: list[tuple[int, str]] = []
    rel = path.relative_to(root).as_posix()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        classes = [item for item in classes if indent > item[0]]

        cls = PY_CLASS.match(line)
        if cls:
            classes.append((len(cls.group(1)), cls.group(2)))
            continue

        func = PY_DEF.match(line)
        if not func:
            continue

        fn_indent = len(func.group(1))
        name = func.group(2)
        if classes:
            cls_indent, cls_name = classes[-1]
            if fn_indent == cls_indent + 4:
                out.append(Symbol(rel, line_no, "method", name, canon(name), cls_name))
            continue

        if fn_indent == 0:
            out.append(Symbol(rel, line_no, "function", name, canon(name)))

    return out


def rust_symbols(path: Path) -> RustFile:
    if not path.exists():
        return RustFile(False, set(), {}, set())

    top: set[str] = set()
    methods: dict[str, set[str]] = {}
    types: set[str] = set()
    current: str | None = None
    depth = 0
    text = remove_rust_tests(path.read_text())

    for line in text.splitlines():
        started = False
        item = RS_TYPE.match(line)
        if item:
            types.add(item.group(1))

        if current is None:
            impl = RS_IMPL.match(line)
            trait = RS_TRAIT.match(line)
            if impl:
                current = strip_impl_type(impl.group(1))
                methods.setdefault(current, set())
                depth = line.count("{") - line.count("}")
                started = True
            elif trait:
                current = trait.group(1)
                methods.setdefault(current, set())
                depth = line.count("{") - line.count("}")
                started = True

        func = RS_FN.match(line)
        if func:
            name = func.group(1)
            if current:
                methods.setdefault(current, set()).add(name)
            else:
                top.add(name)

        if current is not None and not started:
            depth += line.count("{") - line.count("}")
        if current is not None and depth <= 0:
            current = None
            depth = 0

    return RustFile(True, top, methods, types)


def expected_rust(path: Path, py_root: Path, rs_root: Path, mappings: dict[str, str]) -> Path:
    rel = path.relative_to(py_root).as_posix()
    if rel in mappings:
        return rs_root / mappings[rel]
    if rel == "__init__.py":
        return rs_root / "lib.rs"
    if rel.endswith("/__init__.py"):
        return rs_root / rel.removesuffix("/__init__.py") / "mod.rs"
    return rs_root / Path(rel).with_suffix(".rs")


def in_scope(file: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "moshi":
        return file.startswith("moshi/")
    if scope == "demo":
        return file.startswith("demo/")
    return not file.startswith("moshi/") and not file.startswith("demo/")


def matches_file(symbol: Symbol, rust: RustFile) -> bool:
    if symbol.kind == "function":
        return symbol.name in rust.top or symbol.canon in rust.top
    type_names = {norm(name): name for name in rust.types | set(rust.methods)}
    target = type_names.get(norm(symbol.cls or ""))
    if not target:
        return False
    methods = rust.methods.get(target, set())
    return symbol.name in methods or symbol.canon in methods


def matches_anywhere(symbol: Symbol, rust_files: list[RustFile]) -> bool:
    if symbol.kind == "function":
        return any(symbol.name in item.top or symbol.canon in item.top for item in rust_files)

    for item in rust_files:
        type_names = {norm(name): name for name in item.types | set(item.methods)}
        target = type_names.get(norm(symbol.cls or ""))
        if not target:
            continue
        methods = item.methods.get(target, set())
        if symbol.name in methods or symbol.canon in methods:
            return True
    return False


def build_report(args: argparse.Namespace) -> tuple[dict, str]:
    py_root = args.python_root.resolve()
    rs_root = args.rust_root.resolve()
    mappings = {"moshi/models/loaders.py": "loader.rs"}
    for item in args.map:
        source, target = item.split("=", 1)
        mappings[source] = target

    py_files = sorted(py_root.rglob("*.py"))
    rs_cache: dict[Path, RustFile] = {}
    rs_all = [rust_symbols(path) for path in sorted(rs_root.rglob("*.rs"))]
    rows = []

    for py in py_files:
        rel = py.relative_to(py_root).as_posix()
        if not in_scope(rel, args.scope):
            continue
        rs = expected_rust(py, py_root, rs_root, mappings)
        if rs not in rs_cache:
            rs_cache[rs] = rust_symbols(rs)
        rust = rs_cache[rs]

        symbols = python_symbols(py, py_root)
        misses = []
        covered = 0
        for symbol in symbols:
            hit = matches_anywhere(symbol, rs_all) if args.match == "anywhere" else matches_file(symbol, rust)
            if hit:
                covered += 1
                continue
            misses.append(
                {
                    "name": symbol.qual,
                    "line": symbol.line,
                    "kind": symbol.kind,
                }
            )

        rows.append(
            {
                "python": rel,
                "rust": rs.relative_to(rs_root).as_posix() if str(rs).startswith(str(rs_root)) else str(rs),
                "target_exists": rust.exists,
                "total": len(symbols),
                "covered": covered,
                "missing": len(misses),
                "missing_symbols": misses,
            }
        )

    total = sum(row["total"] for row in rows)
    missing = sum(row["missing"] for row in rows)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_root": str(py_root),
        "rust_root": str(rs_root),
        "scope": args.scope,
        "match": args.match,
        "total": total,
        "covered": total - missing,
        "missing": missing,
        "files": rows,
    }
    return data, render(data, args)


def render(data: dict, args: argparse.Namespace) -> str:
    lines = [
        "# Python -> Rust Symbol Gap Report",
        "",
        f"- Python root: `{data['python_root']}`",
        f"- Rust root: `{data['rust_root']}`",
        f"- Scope: `{data['scope']}`",
        f"- Match mode: `{data['match']}`",
        f"- Symbols: {data['covered']}/{data['total']} covered, {data['missing']} missing",
        "",
        "## Missing Symbols",
        "",
    ]

    missing_files = [row for row in data["files"] if row["missing"]]
    if not missing_files:
        lines.append("No missing symbols found.")
    for row in missing_files:
        status = "target exists" if row["target_exists"] else "target missing"
        lines.append(
            f"### `{row['python']}` -> `{row['rust']}` "
            f"({row['covered']}/{row['total']} covered, {status})"
        )
        for symbol in row["missing_symbols"]:
            lines.append(f"- `{symbol['name']}` line {symbol['line']}")
        lines.append("")

    if args.show_covered:
        lines.extend(["## Fully Covered Files", ""])
        covered_files = [row for row in data["files"] if row["total"] and not row["missing"]]
        if not covered_files:
            lines.append("No fully covered files in this scope.")
        for row in covered_files:
            lines.append(f"- `{row['python']}` -> `{row['rust']}` ({row['covered']}/{row['total']})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, default=DEFAULT_PY, help="upstream Python package root")
    parser.add_argument("--rust-root", type=Path, default=DEFAULT_RS, help="Rust src root")
    parser.add_argument("--scope", choices=["core", "all", "moshi", "demo"], default="core")
    parser.add_argument("--match", choices=["file", "anywhere"], default="file")
    parser.add_argument("--map", action="append", default=[], metavar="PY=RS", help="extra Python-to-Rust file mapping")
    parser.add_argument("--json", type=Path, help="write machine-readable report to this path")
    parser.add_argument("--output", type=Path, help="write Markdown report to this path instead of stdout")
    parser.add_argument("--show-covered", action="store_true", help="include fully covered files in Markdown output")
    parser.add_argument("--fail-on-missing", action="store_true", help="exit 1 when scoped symbols are missing")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    data, markdown = build_report(args)

    if args.json:
        args.json.write_text(json.dumps(data, indent=2) + "\n")
    if args.output:
        args.output.write_text(markdown)
    else:
        print(markdown, end="")

    if args.fail_on_missing and data["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

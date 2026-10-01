#!/usr/bin/env python3
"""Assert that an lcov report actually covers the given source files.

Why this exists: the Neo4j integration suites *skip themselves* when
`backends_available()` is false (no Neo4j reachable). A skipped suite still
exits 0, so without this guard the CI would happily upload a coverage report
in which every `src/neo4j/*` file sits at 0 % — and we would read that as
"the code is untestable" instead of "the service was not up".

Usage:
    assert_lcov_covers.py <lcov file> <repo-relative path> [<path> ...]

Exit status 0 when every listed file has at least one covered line, 1
otherwise (printing which files failed and their hit counts).

No third-party dependency: lcov is a line-oriented text format.
  SF:<path>        start of a file record
  DA:<line>,<hits> line coverage datum
  end_of_record    end of a file record
"""

from __future__ import annotations

import sys


def covered_lines_by_file(lcov_path: str) -> dict[str, tuple[int, int]]:
    """Map each file in the report to (covered lines, total instrumented lines).

    Several records can describe the same file (one per test binary when the
    runs are merged); their hits are summed per line before counting.
    """
    per_file: dict[str, dict[int, int]] = {}
    current: str | None = None

    with open(lcov_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("SF:"):
                current = line[3:]
                per_file.setdefault(current, {})
            elif line == "end_of_record":
                current = None
            elif line.startswith("DA:") and current is not None:
                datum = line[3:]
                number, _, hits = datum.partition(",")
                try:
                    line_no = int(number)
                    hit_count = int(hits.split(",")[0])
                except ValueError:
                    continue
                lines = per_file[current]
                lines[line_no] = lines.get(line_no, 0) + hit_count

    return {
        path: (sum(1 for h in lines.values() if h > 0), len(lines))
        for path, lines in per_file.items()
    }


def lookup(report: dict[str, tuple[int, int]], wanted: str) -> tuple[int, int] | None:
    """Find a repo-relative path in a report that may hold absolute paths."""
    if wanted in report:
        return report[wanted]
    suffix = "/" + wanted.lstrip("/")
    matches = [counts for path, counts in report.items() if path.endswith(suffix)]
    if not matches:
        return None
    covered = sum(c for c, _ in matches)
    total = sum(t for _, t in matches)
    return covered, total


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(f"usage: {argv[0]} <lcov file> <path> [<path> ...]", file=sys.stderr)
        return 2

    lcov_path, wanted = argv[1], argv[2:]
    report = covered_lines_by_file(lcov_path)

    failures: list[str] = []
    for path in wanted:
        counts = lookup(report, path)
        if counts is None:
            failures.append(f"{path}: absent from {lcov_path}")
            continue
        covered, total = counts
        if covered == 0:
            failures.append(f"{path}: 0/{total} lines covered")
        else:
            print(f"ok  {path}: {covered}/{total} lines covered")

    if failures:
        print(
            f"\n{len(failures)} file(s) are not covered by {lcov_path}.\n"
            "The usual cause is that the Neo4j/Meilisearch services were not\n"
            "reachable, so the integration suites skipped themselves.",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  FAIL {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

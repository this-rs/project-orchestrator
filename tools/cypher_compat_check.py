#!/usr/bin/env python3
"""
Cypher Compatibility Checker for ObrainDB Migration
=====================================================
Extracts ALL Cypher queries from PO source code, categorizes them,
and flags potential incompatibilities with ObrainDB.

Usage:
    python3 tools/cypher_compat_check.py [--src-dir PATH] [--format json|text]

Output:
    - Per-file query inventory
    - Feature usage matrix (UNWIND, vector, aggregations, etc.)
    - Compatibility assessment per query
    - Summary statistics
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict, Counter
from enum import Enum


class Compat(str, Enum):
    OK = "ok"                    # Fully compatible
    ADAPT = "adapt"              # Needs syntax adaptation (e.g., vector API)
    UNKNOWN = "unknown"          # Can't determine statically
    RISK = "risk"                # Potentially incompatible


@dataclass
class CypherQuery:
    file: str
    line: int
    raw: str
    normalized: str
    method_context: str  # enclosing function name
    features: list = field(default_factory=list)
    compat: Compat = Compat.OK
    compat_notes: list = field(default_factory=list)
    is_dynamic: bool = False  # built with format!/concatenation


@dataclass
class FileReport:
    path: str
    query_count: int = 0
    queries: list = field(default_factory=list)
    feature_counts: dict = field(default_factory=dict)


# ─── Cypher Feature Detectors ────────────────────────────────────────────

FEATURE_PATTERNS = [
    ("MATCH",              r'\bMATCH\b'),
    ("OPTIONAL_MATCH",     r'\bOPTIONAL\s+MATCH\b'),
    ("CREATE",             r'\bCREATE\b'),
    ("MERGE",              r'\bMERGE\b'),
    ("DELETE",             r'\b(?:DETACH\s+)?DELETE\b'),
    ("SET",                r'\bSET\b'),
    ("REMOVE",             r'\bREMOVE\b'),
    ("WITH",               r'\bWITH\b'),
    ("UNWIND",             r'\bUNWIND\b'),
    ("FOREACH",            r'\bFOREACH\b'),
    ("RETURN",             r'\bRETURN\b'),
    ("ORDER_BY",           r'\bORDER\s+BY\b'),
    ("SKIP",               r'\bSKIP\b'),
    ("LIMIT",              r'\bLIMIT\b'),
    ("WHERE",              r'\bWHERE\b'),
    ("CASE_WHEN",          r'\bCASE\s+WHEN\b'),
    ("EXISTS_SUBQUERY",    r'\bEXISTS\s*\{'),
    ("CALL_SUBQUERY",      r'\bCALL\s*\{'),
    ("UNION",              r'\bUNION\b'),
    # Aggregations
    ("AGG_count",          r'\bcount\s*\('),
    ("AGG_collect",        r'\bcollect\s*\('),
    ("AGG_sum",            r'\bsum\s*\('),
    ("AGG_avg",            r'\bavg\s*\('),
    ("AGG_min",            r'\bmin\s*\('),
    ("AGG_max",            r'\bmax\s*\('),
    ("AGG_reduce",         r'\breduce\s*\('),
    # Functions
    ("FN_coalesce",        r'\bcoalesce\s*\('),
    ("FN_toString",        r'\btoString\s*\('),
    ("FN_toInteger",       r'\btoInteger\s*\('),
    ("FN_toFloat",         r'\btoFloat\s*\('),
    ("FN_size",            r'\bsize\s*\('),
    ("FN_labels",          r'\blabels\s*\('),
    ("FN_type",            r'\btype\s*\('),
    ("FN_id",              r'\bid\s*\('),
    ("FN_properties",      r'\bproperties\s*\('),
    ("FN_keys",            r'\bkeys\s*\('),
    ("FN_range",           r'\brange\s*\('),
    ("FN_timestamp",       r'\btimestamp\s*\('),
    ("FN_datetime",        r'\bdatetime\s*\('),
    ("FN_duration",        r'\bduration\s*\('),
    ("FN_point",           r'\bpoint\s*\('),
    ("FN_distance",        r'\bdistance\s*\('),
    ("FN_split",           r'\bsplit\s*\('),
    ("FN_replace",         r'\breplace\s*\('),
    ("FN_trim",            r'\btrim\s*\('),
    ("FN_toLower",         r'\btoLower\s*\('),
    ("FN_toUpper",         r'\btoUpper\s*\('),
    ("FN_substring",       r'\bsubstring\s*\('),
    ("FN_left",            r'\bleft\s*\('),
    ("FN_right",           r'\bright\s*\('),
    ("FN_abs",             r'\babs\s*\('),
    ("FN_rand",            r'\brand\s*\('),
    ("FN_head",            r'\bhead\s*\('),
    ("FN_tail",            r'\btail\s*\('),
    ("FN_last",            r'\blast\s*\('),
    ("FN_nodes",           r'\bnodes\s*\('),
    ("FN_relationships",   r'\brelationships\s*\('),
    ("FN_length",          r'\blength\s*\('),
    ("FN_startNode",       r'\bstartNode\s*\('),
    ("FN_endNode",         r'\bendNode\s*\('),
    # Special / risky
    ("VECTOR_QUERY",       r'db\.index\.vector\.queryNodes'),
    ("VECTOR_SET",         r'db\.create\.setNodeVectorProperty'),
    ("VECTOR_INDEX",       r'(?:CREATE|DROP)\s+VECTOR\s+INDEX'),
    ("APOC",               r'\bapoc\.\w+'),
    ("GDS",                r'\bgds\.\w+'),
    ("FULLTEXT_INDEX",     r'db\.index\.fulltext'),
    ("CONSTRAINT",         r'(?:CREATE|DROP)\s+CONSTRAINT'),
    ("INDEX_CREATE",       r'(?:CREATE|DROP)\s+INDEX'),
    # Patterns
    ("VAR_LENGTH_REL",     r'\[[\w:]*\*'),
    ("PATH_VARIABLE",      r'\w+\s*=\s*\('),
    ("LIST_COMPREHENSION", r'\[[\w\s]+IN\b'),
    ("MAP_PROJECTION",     r'\w+\s*\{[\s\w.:,]+\}'),
    ("PARAM",              r'\$\w+'),
    ("DISTINCT",           r'\bDISTINCT\b'),
]

# ─── Compatibility Rules ─────────────────────────────────────────────────

COMPAT_RULES = {
    "VECTOR_QUERY":    (Compat.ADAPT,  "Obrain uses hybrid_search() procedure instead of db.index.vector.queryNodes()"),
    "VECTOR_SET":      (Compat.ADAPT,  "Obrain uses API/procedure for vector properties, not db.create.setNodeVectorProperty()"),
    "VECTOR_INDEX":    (Compat.ADAPT,  "Obrain vector indexes created via API, not CREATE VECTOR INDEX Cypher"),
    "APOC":            (Compat.RISK,   "APOC is Neo4j-specific, not available in Obrain"),
    "GDS":             (Compat.RISK,   "GDS is Neo4j-specific; Obrain has native CALL algorithms instead"),
    "FULLTEXT_INDEX":  (Compat.ADAPT,  "Obrain uses db.text_search() API instead of db.index.fulltext"),
    "EXISTS_SUBQUERY": (Compat.UNKNOWN, "EXISTS subquery support depends on Obrain version — verify"),
    "FN_point":        (Compat.UNKNOWN, "Spatial functions may not be available in Obrain"),
    "FN_distance":     (Compat.UNKNOWN, "Spatial functions may not be available in Obrain"),
}

# Features that are CONFIRMED compatible with Obrain openCypher 9.0
CONFIRMED_OK = {
    "MATCH", "OPTIONAL_MATCH", "CREATE", "MERGE", "DELETE", "SET", "REMOVE",
    "WITH", "UNWIND", "FOREACH", "RETURN", "ORDER_BY", "SKIP", "LIMIT",
    "WHERE", "CASE_WHEN", "CALL_SUBQUERY", "UNION", "DISTINCT",
    "AGG_count", "AGG_collect", "AGG_sum", "AGG_avg", "AGG_min", "AGG_max",
    "AGG_reduce", "PARAM", "VAR_LENGTH_REL",
    "FN_coalesce", "FN_toString", "FN_toInteger", "FN_toFloat", "FN_size",
    "FN_labels", "FN_type", "FN_id", "FN_properties", "FN_keys", "FN_range",
    "FN_timestamp", "FN_split", "FN_replace", "FN_trim", "FN_toLower",
    "FN_toUpper", "FN_substring", "FN_left", "FN_right", "FN_abs", "FN_rand",
    "FN_head", "FN_tail", "FN_last", "FN_nodes", "FN_relationships",
    "FN_length", "FN_startNode", "FN_endNode",
    "CONSTRAINT", "INDEX_CREATE", "LIST_COMPREHENSION", "MAP_PROJECTION",
    "PATH_VARIABLE",
}


# ─── Cypher Extractor ────────────────────────────────────────────────────

def extract_queries_from_file(filepath: str) -> list[CypherQuery]:
    """Extract Cypher query strings from a Rust source file."""
    queries = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        lines = content.split('\n')

    rel_path = filepath

    # Pattern 1: Raw strings r#"..."# (most common in PO)
    raw_string_pattern = re.compile(r'r#"(.*?)"#', re.DOTALL)

    # Pattern 2: Regular strings with Cypher keywords
    regular_string_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"')

    # Find enclosing function for a position
    def find_method(pos: int) -> str:
        fn_pattern = re.compile(r'\b(?:pub\s+)?(?:async\s+)?fn\s+(\w+)')
        last_fn = "<unknown>"
        for m in fn_pattern.finditer(content[:pos]):
            last_fn = m.group(1)
        return last_fn

    def line_number(pos: int) -> int:
        return content[:pos].count('\n') + 1

    def is_cypher(s: str) -> bool:
        """Check if a string looks like a Cypher query."""
        cypher_keywords = r'\b(MATCH|CREATE|MERGE|RETURN|DELETE|SET|REMOVE|UNWIND|CALL|WITH|OPTIONAL)\b'
        return bool(re.search(cypher_keywords, s, re.IGNORECASE))

    def is_dynamic_context(pos: int) -> bool:
        """Check if the query is inside a format!() or string concat."""
        before = content[max(0, pos-200):pos]
        return bool(re.search(r'format!\s*\(', before)) or '{}' in content[pos:pos+500]

    # Extract from raw strings
    for m in raw_string_pattern.finditer(content):
        raw = m.group(1).strip()
        if is_cypher(raw) and len(raw) > 20:
            queries.append(CypherQuery(
                file=rel_path,
                line=line_number(m.start()),
                raw=raw,
                normalized=normalize_cypher(raw),
                method_context=find_method(m.start()),
                is_dynamic=is_dynamic_context(m.start()),
            ))

    # Extract from regular strings (catch shorter queries)
    for m in regular_string_pattern.finditer(content):
        raw = m.group(1).strip()
        if is_cypher(raw) and len(raw) > 30:
            # Skip if already captured as raw string
            already = any(raw in q.raw for q in queries)
            if not already:
                queries.append(CypherQuery(
                    file=rel_path,
                    line=line_number(m.start()),
                    raw=raw,
                    normalized=normalize_cypher(raw),
                    method_context=find_method(m.start()),
                    is_dynamic=is_dynamic_context(m.start()),
                ))

    # Detect features and compatibility for each query
    for q in queries:
        detect_features(q)
        assess_compat(q)

    return queries


def normalize_cypher(raw: str) -> str:
    """Normalize whitespace in a Cypher query."""
    return re.sub(r'\s+', ' ', raw).strip()


def detect_features(q: CypherQuery):
    """Detect which Cypher features a query uses."""
    for name, pattern in FEATURE_PATTERNS:
        if re.search(pattern, q.raw, re.IGNORECASE):
            q.features.append(name)


def assess_compat(q: CypherQuery):
    """Assess compatibility with Obrain."""
    worst = Compat.OK

    for feat in q.features:
        if feat in COMPAT_RULES:
            level, note = COMPAT_RULES[feat]
            q.compat_notes.append(f"[{feat}] {note}")
            if level == Compat.RISK:
                worst = Compat.RISK
            elif level == Compat.ADAPT and worst != Compat.RISK:
                worst = Compat.ADAPT
            elif level == Compat.UNKNOWN and worst == Compat.OK:
                worst = Compat.UNKNOWN

    if q.is_dynamic:
        q.compat_notes.append("[DYNAMIC] Query built with format!/concat — verify generated Cypher at runtime")
        if worst == Compat.OK:
            worst = Compat.UNKNOWN

    q.compat = worst


# ─── Report Generation ───────────────────────────────────────────────────

def generate_report(all_queries: list[CypherQuery], src_dir: str) -> dict:
    """Generate the full compatibility report."""

    # Group by file
    by_file = defaultdict(list)
    for q in all_queries:
        by_file[q.file].append(q)

    # Feature usage across all queries
    feature_counter = Counter()
    for q in all_queries:
        for f in q.features:
            feature_counter[f] += 1

    # Compat summary
    compat_counter = Counter(q.compat.value for q in all_queries)

    # Queries needing adaptation
    needs_adapt = [q for q in all_queries if q.compat in (Compat.ADAPT, Compat.RISK)]

    # Dynamic queries
    dynamic_queries = [q for q in all_queries if q.is_dynamic]

    # Per-file reports
    file_reports = []
    for fpath, queries in sorted(by_file.items()):
        fc = Counter()
        for q in queries:
            for f in q.features:
                fc[f] += 1
        file_reports.append(FileReport(
            path=fpath,
            query_count=len(queries),
            queries=queries,
            feature_counts=dict(fc),
        ))

    return {
        "summary": {
            "total_queries": len(all_queries),
            "total_files": len(by_file),
            "compatibility": {
                "ok": compat_counter.get("ok", 0),
                "adapt": compat_counter.get("adapt", 0),
                "unknown": compat_counter.get("unknown", 0),
                "risk": compat_counter.get("risk", 0),
            },
            "dynamic_queries": len(dynamic_queries),
            "pct_compatible": round(
                compat_counter.get("ok", 0) / max(len(all_queries), 1) * 100, 1
            ),
            "pct_needs_adapt": round(
                compat_counter.get("adapt", 0) / max(len(all_queries), 1) * 100, 1
            ),
        },
        "feature_usage": dict(feature_counter.most_common()),
        "needs_adaptation": [
            {
                "file": q.file,
                "line": q.line,
                "method": q.method_context,
                "compat": q.compat.value,
                "notes": q.compat_notes,
                "query_preview": q.normalized[:120] + ("..." if len(q.normalized) > 120 else ""),
            }
            for q in needs_adapt
        ],
        "file_reports": [
            {
                "path": fr.path,
                "query_count": fr.query_count,
                "features": fr.feature_counts,
            }
            for fr in sorted(file_reports, key=lambda x: -x.query_count)
        ],
    }


def print_text_report(report: dict):
    """Print a human-readable report."""
    s = report["summary"]

    print("=" * 70)
    print("  CYPHER COMPATIBILITY REPORT — PO → ObrainDB")
    print("=" * 70)
    print()

    # Summary
    print(f"  📊 Total queries extracted:  {s['total_queries']}")
    print(f"  📁 Files scanned:            {s['total_files']}")
    print(f"  🔧 Dynamic queries (format!): {s['dynamic_queries']}")
    print()

    ok = s['compatibility']['ok']
    adapt = s['compatibility']['adapt']
    unknown = s['compatibility']['unknown']
    risk = s['compatibility']['risk']
    total = s['total_queries']

    bar_width = 50
    ok_w = int(ok / max(total, 1) * bar_width)
    adapt_w = int(adapt / max(total, 1) * bar_width)
    unknown_w = int(unknown / max(total, 1) * bar_width)
    risk_w = bar_width - ok_w - adapt_w - unknown_w

    print("  COMPATIBILITY")
    print(f"  [{'█' * ok_w}{'▓' * adapt_w}{'░' * unknown_w}{'▒' * max(risk_w, 0)}]")
    print(f"   ✅ Compatible:    {ok:>4}  ({s['pct_compatible']}%)")
    print(f"   🔧 Needs adapt:   {adapt:>4}  ({s['pct_needs_adapt']}%)")
    print(f"   ❓ Unknown:       {unknown:>4}")
    print(f"   ⚠️  Risk:          {risk:>4}")
    print()

    # Feature usage top 20
    print("  TOP CYPHER FEATURES USED")
    print("  " + "-" * 45)
    for feat, count in list(report["feature_usage"].items())[:25]:
        compat_mark = "✅"
        if feat in COMPAT_RULES:
            level = COMPAT_RULES[feat][0]
            compat_mark = {"adapt": "🔧", "risk": "⚠️ ", "unknown": "❓"}[level.value]
        elif feat in CONFIRMED_OK:
            compat_mark = "✅"
        else:
            compat_mark = "❓"
        print(f"  {compat_mark} {feat:<25} {count:>4} queries")
    print()

    # Files by query count
    print("  QUERIES PER FILE")
    print("  " + "-" * 45)
    for fr in report["file_reports"][:15]:
        short = fr["path"].split("src/")[-1] if "src/" in fr["path"] else fr["path"].split("/")[-1]
        print(f"  {fr['query_count']:>4}  {short}")
    print()

    # Queries needing adaptation
    if report["needs_adaptation"]:
        print(f"  ⚠️  QUERIES NEEDING ADAPTATION ({len(report['needs_adaptation'])})")
        print("  " + "-" * 60)
        for item in report["needs_adaptation"]:
            short_file = item["file"].split("src/")[-1] if "src/" in item["file"] else item["file"].split("/")[-1]
            icon = "🔧" if item["compat"] == "adapt" else "⚠️ "
            print(f"  {icon} {short_file}:{item['line']} → {item['method']}()")
            for note in item["notes"]:
                print(f"       {note}")
            print(f"       Query: {item['query_preview']}")
            print()

    print("=" * 70)
    print(f"  VERDICT: {s['pct_compatible']}% directement compatible,")
    print(f"           {s['pct_needs_adapt']}% nécessite une adaptation syntaxique")
    if risk == 0:
        print(f"           0 requête bloquante (pas d'APOC, pas de GDS)")
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cypher Compatibility Checker for ObrainDB Migration")
    parser.add_argument("--src-dir", default=".", help="PO project root directory")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--json-out", help="Write JSON report to file")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()

    # Scan directories
    scan_dirs = [
        src_dir / "src" / "neo4j",
        src_dir / "crates" / "neural-routing-core" / "src",
        src_dir / "crates" / "neural-routing-gnn" / "src",
    ]

    all_queries = []

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            print(f"  ⚠️  Directory not found: {scan_dir}", file=sys.stderr)
            continue
        for rs_file in sorted(scan_dir.rglob("*.rs")):
            queries = extract_queries_from_file(str(rs_file))
            all_queries.extend(queries)

    if not all_queries:
        print("No Cypher queries found. Check --src-dir path.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(all_queries, str(src_dir))

    if args.format == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        print_text_report(report)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  JSON report saved to: {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()

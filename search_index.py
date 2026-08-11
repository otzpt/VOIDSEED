#!/usr/bin/env python3
"""Full-text search over repos/, for chat.py to call at query time.

This is retrieval, not training data. prepare.py's `security` dataset is
deliberately narrow (prose only, wordlists excluded — see the comment on
LOCAL_RAW_EXTENSIONS there) because raw wordlists would dominate the training
signal 100:1. None of that applies here: a wordlist chunk is only ever
returned when a query actually matches it, so broader coverage is strictly
better for lookup. This indexes every text-bearing file under repos/,
including .txt.

The two halves:

    build_index(repos_dir, db_path)   -- (re)build the index. Run this
                                          whenever repos/ changes.
    search(query, k, db_path)         -- look something up. This is the
                                          function chat.py imports.

Usage:
    python search_index.py                       # build ./search.db
    python search_index.py --query "sql injection" -k 3   # try it
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sqlite3
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "search.db"

# Broader than prepare.py's training-corpus allowlist on purpose: .txt is
# included, because SecLists/PayloadsAllTheThings' raw lists are exactly the
# kind of thing worth being able to look up ("give me common admin
# passwords"), even though they would be bad *training* signal.
INDEX_EXTENSIONS = {".md", ".rst", ".adoc", ".yml", ".yaml", ".txt"}

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def split_by_heading(text: str) -> list[tuple[str, str]]:
    """Return [(heading, section_text), ...].

    Text before the first heading, or a file with no headings at all (a
    wordlist, a YAML file), gets heading "".
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end]))
    return sections


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split into passages of roughly max_chars.

    Breaks on blank lines where possible, so a passage stays one coherent
    paragraph rather than an arbitrary character-count slice. Falls back to a
    hard split by line when a single paragraph exceeds max_chars on its own
    -- the case that matters here is a wordlist file, which is thousands of
    lines with no blank line anywhere.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        if current:
            chunks.append("\n\n".join(current))
            current.clear()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            flush()
            current_len = 0
            buf: list[str] = []
            buf_len = 0
            for line in para.split("\n"):
                if buf_len + len(line) > max_chars and buf:
                    chunks.append("\n".join(buf))
                    buf, buf_len = [], 0
                buf.append(line)
                buf_len += len(line) + 1
            if buf:
                chunks.append("\n".join(buf))
            continue

        if current_len + len(para) > max_chars and current:
            flush()
            current_len = 0
        current.append(para)
        current_len += len(para)

    flush()
    return chunks


def build_index(repos_dir: pathlib.Path, db_path: pathlib.Path,
                max_chars: int = 800) -> int:
    """(Re)build the FTS5 index from everything under repos_dir.

    Drops and recreates the table every time, so the index is always exactly
    in sync with whatever is in repos_dir right now rather than accumulating
    stale passages from files that were since removed. Returns the number of
    passages indexed.
    """
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS passages")
    con.execute("CREATE VIRTUAL TABLE passages USING fts5(repo, path, heading, text)")

    count = 0
    for path in sorted(repos_dir.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in INDEX_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        repo_name = path.relative_to(repos_dir).parts[0]
        rel_path = str(path.relative_to(repos_dir))

        rows = []
        for heading, section in split_by_heading(text):
            for chunk in chunk_text(section, max_chars):
                chunk = chunk.strip()
                if chunk:
                    rows.append((repo_name, rel_path, heading, chunk))

        if rows:
            con.executemany(
                "INSERT INTO passages (repo, path, heading, text) VALUES (?, ?, ?, ?)",
                rows,
            )
            count += len(rows)

    con.commit()
    con.close()
    return count


@dataclass(frozen=True)
class Passage:
    repo: str      # e.g. "hacktricks"
    path: str      # path relative to repos/, e.g. "src/network-services-pentesting/..."
    heading: str   # nearest enclosing markdown heading, "" if none
    text: str      # the passage itself, ~800 chars or less
    score: float   # FTS5 bm25: LOWER is a better match (it's a cost, not a similarity)


def _fts5_query(raw: str) -> str:
    """Turn free-text input into a safe FTS5 MATCH expression.

    FTS5's query syntax gives meaning to -, ", (, ), : and *, so passing a
    natural-language question straight to MATCH can raise a syntax error
    (e.g. a query containing "sql-injection" or a stray quote). Quoting each
    token individually neutralizes that. Terms are OR'd rather than AND'd:
    for a lookup tool, recall matters more than precision, and bm25 ranking
    still surfaces the passages matching the most terms first.
    """
    terms = re.findall(r"\w+", raw)
    if not terms:
        return '""'
    return " OR ".join(f'"{t}"' for t in terms)


def search(query: str, k: int = 5,
          db_path: pathlib.Path = DEFAULT_DB_PATH) -> list[Passage]:
    """Look up passages relevant to `query`. Best match first.

    Opens and closes its own connection per call. Fine for a chat loop
    calling this a few times per turn; if you end up calling it in a tight
    loop, open one connection yourself and reuse the query instead.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT repo, path, heading, text, bm25(passages) AS score
            FROM passages
            WHERE passages MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (_fts5_query(query), k),
        ).fetchall()
    finally:
        con.close()
    return [Passage(r["repo"], r["path"], r["heading"], r["text"], r["score"])
            for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", type=pathlib.Path, default=ROOT / "repos")
    ap.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--query", help="skip (re)building; just search the existing index")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.query:
        for p in search(args.query, k=args.k, db_path=args.db):
            print(f"[{p.score:.2f}] {p.repo}/{p.path}"
                  f"{'  > ' + p.heading if p.heading else ''}")
            preview = p.text[:200].replace("\n", " ")
            print(f"    {preview}...")
        return 0

    n = build_index(args.repos_dir, args.db)
    print(f"indexed {n:,} passages -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

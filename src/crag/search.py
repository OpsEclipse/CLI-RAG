from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np

from crag.embeddings import cosine_similarity, deserialize_vector
from crag.models import SearchResult

_FTS_TERM_RE = re.compile(r"\w+", re.UNICODE)


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    low = min(scores.values())
    high = max(scores.values())
    if low == high:
        return {chunk_id: 1.0 for chunk_id in scores}

    return {
        chunk_id: (score - low) / (high - low) for chunk_id, score in scores.items()
    }


def build_fts_query(query: str) -> str | None:
    terms = _FTS_TERM_RE.findall(query)
    if not terms:
        return None

    quoted_terms = []
    for term in terms:
        escaped = term.replace('"', '""')
        quoted_terms.append(f'"{escaped}"')
    return " OR ".join(quoted_terms)


def snippet(text: str, query: str, limit: int = 120) -> str:
    clean_text = " ".join(text.split())
    if len(clean_text) <= limit:
        return clean_text

    terms = [term for term in re.split(r"\W+", query.lower()) if term]
    match_index = -1
    lower_text = clean_text.lower()
    for term in terms:
        match_index = lower_text.find(term)
        if match_index >= 0:
            break

    if match_index < 0:
        return clean_text[: limit - 3].rstrip() + "..."

    start = max(0, match_index - limit // 3)
    end = min(len(clean_text), start + limit)
    start = max(0, end - limit)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    body_limit = limit - len(prefix) - len(suffix)
    return prefix + clean_text[start : start + body_limit].strip() + suffix


def result_from_chunk(
    conn: sqlite3.Connection,
    chunk_id: int,
    score: float,
    number: int,
    query: str,
) -> SearchResult:
    row = conn.execute(
        """
        SELECT
            chunks.id,
            chunks.text,
            chunks.topic,
            chunks.location,
            documents.path,
            documents.file_name
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Chunk not found: {chunk_id}")

    return SearchResult(
        result_number=number,
        chunk_id=int(row["id"]),
        file_path=Path(row["path"]),
        file_name=str(row["file_name"]),
        location=str(row["location"]),
        topic=str(row["topic"]),
        snippet=snippet(str(row["text"]), query),
        score=score,
    )


def keyword_scores(
    conn: sqlite3.Connection,
    query: str,
    file_filter: str | None = None,
    top: int = 20,
) -> dict[int, float]:
    fts_query = build_fts_query(query)
    if fts_query is None:
        return {}

    params: list[object] = [fts_query]
    file_clause = ""
    if file_filter:
        file_clause = "AND documents.file_name LIKE ?"
        params.append(f"%{file_filter}%")
    params.append(top)

    rows = conn.execute(
        f"""
        SELECT chunks.id AS chunk_id, bm25(chunk_fts) * -1 AS score
        FROM chunk_fts
        JOIN chunks ON chunks.id = chunk_fts.rowid
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunk_fts MATCH ?
        {file_clause}
        ORDER BY score DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {int(row["chunk_id"]): float(row["score"]) for row in rows}


def semantic_scores(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    file_filter: str | None = None,
    top: int = 20,
) -> dict[int, float]:
    query_vector = np.asarray(query_vector, dtype=np.float32)
    params: list[object] = []
    file_clause = ""
    if file_filter:
        file_clause = "WHERE documents.file_name LIKE ?"
        params.append(f"%{file_filter}%")

    rows = conn.execute(
        f"""
        SELECT embeddings.chunk_id, embeddings.vector
        FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        {file_clause}
        """,
        params,
    ).fetchall()

    scores: dict[int, float] = {}
    for row in rows:
        try:
            vector = deserialize_vector(
                row["vector"], expected_dimensions=int(query_vector.size)
            )
        except ValueError:
            continue
        scores[int(row["chunk_id"])] = cosine_similarity(query_vector, vector)

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top])


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    top: int = 5,
    file_filter: str | None = None,
) -> list[SearchResult]:
    scores = keyword_scores(conn, query, file_filter=file_filter, top=top)
    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(scores.items(), start=1)
    ]


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    query_vector: np.ndarray,
    top: int = 5,
    file_filter: str | None = None,
) -> list[SearchResult]:
    scores = semantic_scores(conn, query_vector, file_filter=file_filter, top=top)
    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(scores.items(), start=1)
    ]


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_vector: np.ndarray,
    alpha: float = 0.5,
    top: int = 5,
    file_filter: str | None = None,
) -> list[SearchResult]:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    candidate_limit = max(top * 4, 20)
    keyword = normalize_scores(
        keyword_scores(conn, query, file_filter=file_filter, top=candidate_limit)
    )
    semantic = normalize_scores(
        semantic_scores(
            conn, query_vector, file_filter=file_filter, top=candidate_limit
        )
    )
    chunk_ids = set(keyword) | set(semantic)
    final_scores = {
        chunk_id: alpha * semantic.get(chunk_id, 0.0)
        + (1 - alpha) * keyword.get(chunk_id, 0.0)
        for chunk_id in chunk_ids
    }
    ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:top]

    return [
        result_from_chunk(conn, chunk_id, score, number, query)
        for number, (chunk_id, score) in enumerate(ranked, start=1)
    ]


def save_last_search(
    conn: sqlite3.Connection,
    results: list[SearchResult],
    mode: str,
) -> None:
    conn.execute("DELETE FROM last_search_results")
    conn.executemany(
        """
        INSERT INTO last_search_results(result_number, chunk_id, mode, score)
        VALUES (?, ?, ?, ?)
        """,
        [
            (result.result_number, result.chunk_id, mode, result.score)
            for result in results
        ],
    )
    conn.commit()

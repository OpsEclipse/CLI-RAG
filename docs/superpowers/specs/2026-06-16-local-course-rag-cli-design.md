# Local Course Search CLI Design

## Purpose

Build a local command-line tool named `crag` for fast search across course materials during an open-book exam.

The tool may use the internet during ingestion. Ingestion is the setup step where files are parsed and indexed. Querying must work offline after ingestion.

The tool must not generate answers, summaries, explanations, or rewritten content during the exam. It only finds and points to source material.

## User Goal

The user wants to download slides, PDFs, and related course content before an exam. During the exam, the user can search locally for terms, concepts, and phrases. The CLI returns the most relevant source locations, such as file path, slide number, page number, topic, and matching text.

This should feel like a fast, course-specific index at the back of a textbook. It should not behave like a tutor or chatbot.

## Constraints

- Ingestion may use Mistral OCR API.
- Search must work without internet after ingestion.
- No LLM answer generation in search.
- No generated summaries in search.
- No evidence HTML pages in version 1.
- `crag open` opens the original source file.
- Results should display in polished terminal tables.

## Supported Content

Version 1 supports files that Mistral OCR can parse, including:

- PDF files.
- PPTX files.
- DOCX files.
- Image files supported by Mistral OCR.

The default scanner should include these extensions:

- `.pdf`
- `.pptx`
- `.docx`
- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

Unsupported files should be skipped and counted in the ingest summary.

## CLI Commands

### Ingest

```bash
crag ingest ./course-materials
```

Recursively scans a folder, sends supported files to Mistral OCR, and stores parsed content locally.

Ingestion stores:

- Original file path.
- File type.
- Slide number or page number when available.
- Topic or heading.
- OCR text.
- Searchable chunks.
- Raw OCR response.
- Ingest timestamp.
- Status and warnings.

Raw OCR responses are saved so the local index can be rebuilt without another API call.

### Search

```bash
crag search "price sensitivity"
```

Default search mode is hybrid.

Hybrid combines keyword and semantic scores:

```text
final_score = alpha * semantic_score + (1 - alpha) * keyword_score
```

Default `alpha` is `0.5`.

Before combining, scores must be normalized. Normalization means rescaling scores to the same range. This is required because BM25 keyword scores and semantic similarity scores use different scales.

Search options:

```bash
crag search "price sensitivity" --keyword
crag search "price sensitivity" --semantic
crag search "price sensitivity" --alpha 0.7
crag search "elasticity" --top 10
crag search "elasticity" --file week-03
```

Mode behavior:

- Default: hybrid ranked results.
- `--keyword`: keyword-only results.
- `--semantic`: semantic-only results.
- `--alpha`: adjusts hybrid weighting. Values closer to `1.0` favor semantic search. Values closer to `0.0` favor keyword search.

`--alpha` must be between `0.0` and `1.0`.

Search must not call any internet service.

### Open

```bash
crag open 1
```

Opens the original file for result `1` from the most recent search.

For PowerPoint files, `crag open` opens the file and prints the slide number.

For PDF files, `crag open` should try to open the page directly when the operating system supports it. It must still print the page number either way.

Example output:

```text
Opened: /course/week-04.pptx
Go to: Slide 18
Topic: Elasticity
Match: "price elasticity measures responsiveness..."
```

### Status

```bash
crag status
```

Shows index totals and readiness.

Example:

```text
Index Status

Documents:      42
Slides/pages:   1,284
Text chunks:    3,912
Last ingested:  2026-06-16 20:42
Semantic index: Available
Search online:  No
```

### List

```bash
crag list
crag list --errors
```

Shows ingested files in a table. `--errors` shows only failed or warning files.

## Result Display

Use Rich for terminal tables.

Default hybrid output uses one ranked table.

Example:

```text
Hybrid Results

┏━━━┳━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ # ┃ File         ┃ Loc ┃ Topic      ┃ Match               ┃ Score ┃
┣━━━╋━━━━━━━━━━━━━━╋━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━┫
┃ 1 ┃ week-04.pptx ┃ S18 ┃ Elasticity ┃ price elasticity... ┃ 0.92  ┃
┗━━━┻━━━━━━━━━━━━━━┻━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┛
```

Keyword-only output uses `Keyword Results`.

Semantic-only output uses `Semantic Results`.

Columns:

- Result number.
- File name.
- Location, such as `S18` or `P42`.
- Topic.
- Matching snippet.
- Score.

The result number is used by `crag open`.

## Topic Extraction

No AI-generated topics in version 1.

Topic should come from source structure:

- Slide title when available.
- PDF heading when available.
- First strong heading in OCR markdown.
- `Untitled` if no heading is found.

## Storage Design

Use SQLite for local persistent storage.

SQLite stores:

- Documents.
- Pages or slides.
- Chunks.
- Search metadata.
- Ingest status.
- Last search results.

Use SQLite FTS5 for keyword search. FTS5 is SQLite's built-in full-text search system.

Use a local vector index for semantic search. A vector index stores numeric text fingerprints for meaning-based lookup.

The semantic model must run locally. It must not call the internet during search.

## Indexing Design

Ingestion creates chunks at slide or page boundaries first.

Long pages may be split into smaller chunks. Each chunk keeps a pointer back to:

- Document.
- Page or slide.
- Topic.
- Original text.

This lets search show short snippets while `crag open` still points to the full source location.

## Hybrid Ranking

For a query, hybrid search runs both:

- Keyword search through FTS5 or BM25.
- Semantic search through local embeddings.

Each result gets normalized scores:

- `keyword_score_normalized`
- `semantic_score_normalized`

Then:

```text
final_score = alpha * semantic_score_normalized + (1 - alpha) * keyword_score_normalized
```

If a result appears in only one search type, the missing normalized score is `0`.

Default `alpha` is `0.5`.

## Error Handling

Ingestion should continue if one file fails.

Failures are saved with:

- File path.
- Error type.
- Error message.
- Timestamp.

The user can inspect them with:

```bash
crag list --errors
```

Search should show a clear message if no index exists:

```text
No index found. Run: crag ingest <folder>
```

Search should show a clear message if semantic search is requested before semantic indexing exists:

```text
Semantic index not available. Re-run ingestion with semantic indexing enabled.
```

## Testing Strategy

Tests should cover:

- Ingest metadata storage.
- Raw OCR response storage.
- Chunk creation.
- Topic extraction fallback.
- Keyword search ranking.
- Semantic search mode selection.
- Hybrid score normalization and alpha weighting.
- Result table rendering.
- `crag open` lookup from last search results.
- Error listing.

Integration tests should use fixture OCR responses instead of calling Mistral.

## Out of Scope for Version 1

- Generated answers.
- Generated summaries.
- Chat interface.
- Web UI.
- Evidence HTML pages.
- Cloud search.
- Internet calls during query time.
- Automatic PowerPoint slide jumping.

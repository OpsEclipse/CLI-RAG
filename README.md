# crag

`crag` is a local course search CLI.

It uses Mistral OCR during ingestion only. Ingestion is the setup step where files are parsed and indexed.

Search works offline after ingestion.

`crag` does not generate answers, summaries, or explanations. It only points to source material.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Ingest

```bash
export MISTRAL_API_KEY="your-key"
crag ingest ./course-materials
```

You can also put the key in `.env` or `.env.local`:

```bash
MISTRAL_API_KEY=your-key
```

Ingestion may use the internet.

It sends supported files to Mistral OCR. It also downloads the local embedding model from Hugging Face if it is not already cached.

The embedding model is `BAAI/bge-small-en-v1.5`.

Supported file extensions:

- `.pdf`
- `.pptx`
- `.docx`
- `.png`
- `.jpg`
- `.jpeg`
- `.webp`

## Search

```bash
crag search "price sensitivity"
crag search "price sensitivity" --keyword
crag search "price sensitivity" --semantic
crag search "price sensitivity" --alpha 0.7
crag search "elasticity" --top 10
crag search "elasticity" --file week-03
```

Search must work without internet.

Default search is hybrid. Hybrid search combines keyword search and semantic search [search by meaning using numeric text fingerprints].

```text
final_score = alpha * semantic_score + (1 - alpha) * keyword_score
```

The default alpha is `0.5`.

## Open

```bash
crag open 1
```

This opens the original source file from the most recent search and prints the slide or page location.

## Manage The Index

```bash
crag status
crag list
crag list --errors
crag delete 3
crag delete /path/to/file.pptx
crag delete --all
crag delete --all --yes
```

Delete commands only remove indexed data. They do not delete the original source files.

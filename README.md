# crag

`crag` is a local course search CLI.

It uses Mistral OCR during ingestion only. Ingestion is the setup step where files are parsed and indexed.

Search works offline after ingestion.

`crag` does not generate answers, summaries, or explanations. It only points to source material.

## Planned commands

```bash
crag ingest ./course-materials
crag search "price sensitivity"
crag search "price sensitivity" --keyword
crag search "price sensitivity" --semantic
crag search "price sensitivity" --alpha 0.7
crag open 1
crag status
crag list
crag list --errors
crag delete 3
crag delete --all
```

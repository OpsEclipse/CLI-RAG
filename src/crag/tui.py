from __future__ import annotations

import sqlite3
from typing import Any, Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Select, Static

from crag import config, openers
from crag.db import connect, init_db
from crag.models import SearchResult

SearchMode = Literal["hybrid", "keyword", "semantic"]


class CragTuiApp(App[None]):
    """Interactive local search app."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #controls {
        height: 7;
        padding: 1 1 0 1;
    }

    #search-row {
        height: 3;
    }

    #search-query {
        width: 1fr;
    }

    #mode-select {
        width: 18;
    }

    #file-filter {
        width: 1fr;
    }

    #top-count {
        width: 12;
    }

    #results {
        height: 1fr;
    }

    #status {
        height: 1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("f", "focus_file", "File"),
        Binding("m", "cycle_mode", "Mode"),
        Binding("enter", "open_selected", "Open"),
        Binding("o", "open_selected", "Open"),
        Binding("r", "run_search", "Run"),
        Binding("ctrl+f", "focus_file", "File", show=False, priority=True),
        Binding("ctrl+k", "cycle_mode", "Mode", show=False, priority=True),
        Binding("ctrl+o", "open_selected", "Open", show=False, priority=True),
        Binding("ctrl+r", "run_search", "Run", show=False, priority=True),
    ]

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__()
        self.db_path = config.DB_PATH if db_path is None else db_path
        self.conn: sqlite3.Connection | None = None
        self.embedding_model: Any | None = None
        self.results: list[SearchResult] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="controls"):
            with Horizontal(id="search-row"):
                yield Input(
                    placeholder="Search course materials",
                    id="search-query",
                )
            with Horizontal(id="filters-row"):
                yield Select(
                    [("Hybrid", "hybrid"), ("Keyword", "keyword"), ("Semantic", "semantic")],
                    value="hybrid",
                    allow_blank=False,
                    id="mode-select",
                )
                yield Input(placeholder='File filter, like "week 3"', id="file-filter")
                yield Input(value="5", placeholder="Top", id="top-count")
        yield DataTable(id="results")
        yield Static("Type a search and press Enter. q quits.", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.conn = connect(self.db_path)
        init_db(self.conn)
        table = self.query_one("#results", DataTable)
        table.cursor_type = "row"
        table.add_columns("No.", "Score", "File", "Location", "Topic", "Snippet")
        self.query_one("#search-query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {"search-query", "file-filter", "top-count"}:
            self.action_run_search()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mode-select":
            query = self.query_one("#search-query", Input).value.strip()
            if query:
                self.action_run_search()

    def action_focus_search(self) -> None:
        self.query_one("#search-query", Input).focus()

    def action_focus_file(self) -> None:
        self.query_one("#file-filter", Input).focus()

    def action_cycle_mode(self) -> None:
        mode_select = self.query_one("#mode-select", Select)
        next_mode = self._next_search_mode(mode_select.value)
        mode_select.value = next_mode
        self._set_status(f"Mode: {next_mode}")

    def action_run_search(self) -> None:
        query = self.query_one("#search-query", Input).value.strip()
        if not query:
            self._set_status("Type a search first.")
            return

        try:
            top = int(self.query_one("#top-count", Input).value.strip() or "5")
        except ValueError:
            self._set_status("Top must be a number.")
            return
        if top < 1:
            self._set_status("Top must be at least 1.")
            return

        mode = self._selected_search_mode()
        file_filter = self.query_one("#file-filter", Input).value.strip() or None
        self._set_status(f"Searching {mode}...")

        try:
            self.results = self._search(query, mode, top, file_filter)
        except Exception as exc:
            self.results = []
            self._refresh_table()
            self._set_status(f"Search failed: {exc}")
            return

        self._refresh_table()
        self.set_focus(self.query_one("#results", DataTable))
        count = len(self.results)
        plural = "" if count == 1 else "s"
        self._set_status(f"{count} result{plural}. Enter opens the selected source file.")

    def action_open_selected(self) -> None:
        if not self.results:
            self.action_run_search()
            return

        table = self.query_one("#results", DataTable)
        row_index = table.cursor_row
        if row_index < 0 or row_index >= len(self.results):
            self._set_status("Select a result first.")
            return

        result = self.results[row_index]
        try:
            openers.open_file(result.file_path)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            self._set_status(f"Open failed: {exc}")
            return
        self._set_status(f"Opened {result.file_name}. Go to {result.location}.")

    def _search(
        self,
        query: str,
        mode: SearchMode,
        top: int,
        file_filter: str | None,
    ) -> list[SearchResult]:
        if self.conn is None:
            raise RuntimeError("Database connection is not ready.")

        from crag.search import (
            hybrid_search,
            keyword_search,
            save_last_search,
            semantic_search,
        )

        if mode == "keyword":
            results = keyword_search(self.conn, query, top=top, file_filter=file_filter)
        else:
            from crag.embeddings import embed_texts

            model = self._load_embedding_model()
            query_vector = embed_texts(model, [query])[0]
            if mode == "semantic":
                results = semantic_search(
                    self.conn, query, query_vector, top=top, file_filter=file_filter
                )
            else:
                results = hybrid_search(
                    self.conn,
                    query,
                    query_vector,
                    alpha=0.5,
                    top=top,
                    file_filter=file_filter,
                )

        save_last_search(self.conn, results, mode=mode)
        return results

    def _load_embedding_model(self) -> Any:
        if self.embedding_model is None:
            from crag.embeddings import load_model

            self.embedding_model = load_model(local_only=True)
        return self.embedding_model

    def _refresh_table(self) -> None:
        table = self.query_one("#results", DataTable)
        table.clear()
        for result in self.results:
            table.add_row(
                str(result.result_number),
                f"{result.score:.3f}",
                result.file_name,
                result.location,
                result.topic,
                result.snippet,
            )

    def _selected_search_mode(self) -> SearchMode:
        mode = str(self.query_one("#mode-select", Select).value)
        if mode not in {"hybrid", "keyword", "semantic"}:
            return "hybrid"
        return mode  # type: ignore[return-value]

    def _next_search_mode(self, current_value: object) -> SearchMode:
        modes: list[SearchMode] = ["hybrid", "keyword", "semantic"]
        current = str(current_value)
        if current not in modes:
            return "hybrid"
        return modes[(modes.index(current) + 1) % len(modes)]

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)


def run_tui() -> None:
    CragTuiApp().run()

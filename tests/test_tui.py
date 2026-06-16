import asyncio

import numpy as np
from textual.widgets import DataTable, Input, Select

from crag.db import connect, init_db
from crag.embeddings import serialize_vector
from crag.tui import CragTuiApp


def seed_tui_chunk(db_path):
    conn = connect(db_path)
    init_db(conn)
    document_id = conn.execute(
        """
        INSERT INTO documents(path, file_name, file_type, status)
        VALUES (?, ?, 'pptx', 'ready')
        """,
        (str(db_path.parent / "week-03.pptx"), "week-03.pptx"),
    ).lastrowid
    item_id = conn.execute(
        """
        INSERT INTO items(document_id, item_number, item_kind, topic, text)
        VALUES (?, 1, 'slide', 'Elasticity', 'Price elasticity measures responsiveness.')
        """,
        (document_id,),
    ).lastrowid
    chunk_id = conn.execute(
        """
        INSERT INTO chunks(document_id, item_id, chunk_index, text, topic, location)
        VALUES (?, ?, 0, 'Price elasticity measures responsiveness.', 'Elasticity', 'S1')
        """,
        (document_id, item_id),
    ).lastrowid
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text, topic, file_name) VALUES (?, ?, ?, ?)",
        (
            chunk_id,
            "Price elasticity measures responsiveness.",
            "Elasticity",
            "week-03.pptx",
        ),
    )
    conn.execute(
        "INSERT INTO embeddings(chunk_id, model_name, vector) VALUES (?, 'test-model', ?)",
        (chunk_id, serialize_vector(np.array([1, 0], dtype=np.float32))),
    )
    conn.commit()
    conn.close()


def test_tui_mounts_search_screen(tmp_path):
    async def run_test():
        app = CragTuiApp(db_path=tmp_path / "crag.db")
        async with app.run_test():
            assert app.query_one("#search-query", Input).placeholder == (
                "Search course materials"
            )
            assert app.query_one("#results", DataTable).cursor_type == "row"

    asyncio.run(run_test())


def test_tui_keyword_search_populates_results(tmp_path, monkeypatch):
    async def run_test():
        db_path = tmp_path / "crag.db"
        seed_tui_chunk(db_path)
        monkeypatch.setattr(
            "crag.embeddings.load_model",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("keyword TUI search should not load embeddings")
            ),
        )

        app = CragTuiApp(db_path=db_path)
        async with app.run_test():
            app.query_one("#mode-select", Select).value = "keyword"
            app.query_one("#search-query", Input).value = "elasticity"
            app.action_run_search()

            table = app.query_one("#results", DataTable)
            assert len(app.results) == 1
            assert app.results[0].file_name == "week-03.pptx"
            assert app.results[0].location == "S1"
            assert table.row_count == 1

    asyncio.run(run_test())


def test_tui_focuses_results_after_search(tmp_path):
    async def run_test():
        db_path = tmp_path / "crag.db"
        seed_tui_chunk(db_path)

        app = CragTuiApp(db_path=db_path)
        async with app.run_test() as pilot:
            app.query_one("#mode-select", Select).value = "keyword"
            app.query_one("#search-query", Input).value = "elasticity"
            app.action_run_search()
            await pilot.pause()

            assert app.query_one("#results", DataTable).has_focus

    asyncio.run(run_test())


def test_tui_hybrid_search_loads_embedding_model_local_only(tmp_path, monkeypatch):
    async def run_test():
        db_path = tmp_path / "crag.db"
        seed_tui_chunk(db_path)
        load_calls = []

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                assert texts == ["elasticity"]
                assert normalize_embeddings is True
                return [np.array([1, 0], dtype=np.float32)]

        def fake_load_model(local_only=True):
            load_calls.append(local_only)
            return FakeModel()

        monkeypatch.setattr("crag.embeddings.load_model", fake_load_model)

        app = CragTuiApp(db_path=db_path)
        async with app.run_test():
            app.query_one("#search-query", Input).value = "elasticity"
            app.action_run_search()

            assert load_calls == [True]
            assert len(app.results) == 1

    asyncio.run(run_test())

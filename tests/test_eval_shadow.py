"""The schema rewrite that makes the shadow corpus run the same SQL."""

from __future__ import annotations

from eval.shadow import rewrite


def test_schema_references_are_rewritten_and_lookalikes_left_alone():
    sql = (
        "SELECT c.chunk_id FROM rag.chunks c JOIN rag.features f ON rag.corpus_tsquery(x) "
        "WHERE owner = 'urban_rag.x' AND other = rag_eval.y AND col = t.rag.z"
    )
    out = rewrite(sql, "rag", "rag_eval")
    assert "FROM rag_eval.chunks" in out
    assert "JOIN rag_eval.features" in out
    assert "rag_eval.corpus_tsquery" in out
    assert "'urban_rag.x'" in out
    assert "other = rag_eval.y" in out
    # A qualified column `t.rag.z` is not a schema reference.
    assert "t.rag.z" in out


def test_a_function_definition_keeps_its_own_name_in_the_new_schema():
    definition = "CREATE OR REPLACE FUNCTION rag.search_corpus(query_embedding vector)\n RETURNS TABLE(x int)\nAS $$ SELECT 1 FROM rag.chunks $$"
    out = rewrite(definition, "rag", "rag_eval")
    assert out.startswith("CREATE OR REPLACE FUNCTION rag_eval.search_corpus(")
    assert "FROM rag_eval.chunks" in out

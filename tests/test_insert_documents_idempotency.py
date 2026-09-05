"""
Regression coverage for app.services.ingestion.insert_documents's
idempotency key (POST_LEASE_MVP_SPRINT_PLAN.md item 4). Closes a real
coverage gap: before this file, NOTHING in the suite called
insert_documents directly -- every ingestion test mocks around it
(tests/test_ingestion.py's own comment: "ingest_file (which touches
Postgres via ingest_text/insert_documents) is mocked") -- so a
duplicate-insert regression would have passed the whole suite unchanged.

Requires real Postgres with the current schema (content_hash column +
uq_documents_tenant_content_hash unique index on documents) -- gated on
the existing postgres_reachable fixture, same convention as
tests/test_routing_darija_e2e.py.
"""
import uuid

import pytest

from app.services.ingestion import get_db_connection, insert_documents
from app.config import get_settings

TEST_TENANT = "idempotency_test_tenant"


@pytest.fixture(autouse=True)
def _require_postgres_and_cleanup(postgres_reachable):
    if not postgres_reachable:
        pytest.skip("Postgres not reachable")
    yield
    conn = get_db_connection(get_settings().database_url)
    try:
        cur = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()
        cur.execute("DELETE FROM documents WHERE tenant_id LIKE %s", (f"{TEST_TENANT}%",))
        conn.commit()
    finally:
        conn.close()


def _tiny_embedding() -> list[float]:
    return [0.001] * get_settings().embedding_dim


def _row_count(tenant_id: str) -> int:
    conn = get_db_connection(get_settings().database_url)
    try:
        cur = conn.conn.cursor() if hasattr(conn, "conn") else conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents WHERE tenant_id = %s", (tenant_id,))
        return cur.fetchone()[0]
    finally:
        conn.close()


def test_reingesting_identical_chunks_is_a_noop():
    """The core guarantee: insert the same (tenant, source, content) twice
    -- the second call must insert zero new rows, not a duplicate."""
    tenant = f"{TEST_TENANT}_1"
    conn = get_db_connection(get_settings().database_url)
    try:
        chunks = ["Le port du casque est obligatoire.", "L'employeur doit fournir les EPI."]
        embeddings = [_tiny_embedding(), _tiny_embedding()]
        first = insert_documents(
            conn, tenant, "test_doc.md", "markdown", "fr", chunks, embeddings, domain="industrial",
        )
        assert first == 2
        second = insert_documents(
            conn, tenant, "test_doc.md", "markdown", "fr", chunks, embeddings, domain="industrial",
        )
        assert second == 0, "re-ingesting identical chunks must insert 0 rows, not duplicate them"
    finally:
        conn.close()
    assert _row_count(tenant) == 2


def test_partial_overlap_only_inserts_the_new_chunks():
    """A re-ingest where ONE chunk changed (e.g. a document edit) must
    insert only the new/changed chunk, not skip the whole batch."""
    tenant = f"{TEST_TENANT}_2"
    conn = get_db_connection(get_settings().database_url)
    try:
        first_chunks = ["Chunk A unchanged.", "Chunk B original version."]
        embeddings = [_tiny_embedding(), _tiny_embedding()]
        first = insert_documents(
            conn, tenant, "doc.md", "markdown", "fr", first_chunks, embeddings, domain="industrial",
        )
        assert first == 2

        second_chunks = ["Chunk A unchanged.", "Chunk B REVISED version."]
        second = insert_documents(
            conn, tenant, "doc.md", "markdown", "fr", second_chunks, embeddings, domain="industrial",
        )
        assert second == 1, "only the changed chunk should insert; the unchanged one is a duplicate"
    finally:
        conn.close()
    assert _row_count(tenant) == 3


def test_identical_content_in_different_source_documents_both_insert():
    """The idempotency key is scoped to (tenant, source_file_id-or-source_
    name, content) -- NOT content alone. Identical boilerplate text
    appearing in two different source documents must insert twice, not
    collide as if it were a re-ingest of the same document."""
    tenant = f"{TEST_TENANT}_3"
    conn = get_db_connection(get_settings().database_url)
    try:
        shared_text = ["Cette clause standard apparaît dans plusieurs documents."]
        embeddings = [_tiny_embedding()]
        first = insert_documents(
            conn, tenant, "doc_a.md", "markdown", "fr", shared_text, embeddings, domain="industrial",
        )
        second = insert_documents(
            conn, tenant, "doc_b.md", "markdown", "fr", shared_text, embeddings, domain="industrial",
        )
        assert first == 1
        assert second == 1, "identical content in a DIFFERENT source document must still insert"
    finally:
        conn.close()
    assert _row_count(tenant) == 2


def test_identical_content_in_different_tenants_both_insert():
    """The idempotency key is scoped per-tenant -- two tenants uploading
    the same boilerplate text must not collide with each other."""
    tenant_a = f"{TEST_TENANT}_4a"
    tenant_b = f"{TEST_TENANT}_4b"
    conn = get_db_connection(get_settings().database_url)
    try:
        text = ["Texte identique chez deux tenants differents."]
        embeddings = [_tiny_embedding()]
        a = insert_documents(
            conn, tenant_a, "doc.md", "markdown", "fr", text, embeddings, domain="industrial",
        )
        b = insert_documents(
            conn, tenant_b, "doc.md", "markdown", "fr", text, embeddings, domain="industrial",
        )
        assert a == 1
        assert b == 1
    finally:
        conn.close()
    assert _row_count(tenant_a) == 1
    assert _row_count(tenant_b) == 1


def test_uploaded_source_scoping_uses_source_file_id_not_source_name():
    """Two different uploads (different source_file_id) that happen to
    share a filename must not collide -- the identity key prefers
    source_file_id over source_name exactly for this reason."""
    tenant = f"{TEST_TENANT}_5"
    conn = get_db_connection(get_settings().database_url)
    try:
        text = ["Meme nom de fichier, deux uploads distincts."]
        embeddings = [_tiny_embedding()]
        sfid_a = str(uuid.uuid4())
        sfid_b = str(uuid.uuid4())
        a = insert_documents(
            conn, tenant, "report.pdf", "pdf", "fr", text, embeddings,
            domain="industrial", source_file_id=sfid_a,
        )
        b = insert_documents(
            conn, tenant, "report.pdf", "pdf", "fr", text, embeddings,
            domain="industrial", source_file_id=sfid_b,
        )
        assert a == 1
        assert b == 1, "different source_file_id (even with the same filename) must both insert"
    finally:
        conn.close()
    assert _row_count(tenant) == 2

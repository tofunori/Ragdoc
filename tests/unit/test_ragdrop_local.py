"""Local mode boundaries: explicit paths, no inherited keys, no accidental broad writes."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('ragdrop_local', Path(__file__).parents[2] / 'scripts/ragdrop_local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def test_configuration_overrides_inherited_remote_and_private_environment(monkeypatch, tmp_path):
    env = dict(os.environ, VOYAGE_API_KEY='must-not-inherit', COHERE_API_KEY='must-not-inherit',
               RAGDOC_CHROMA_MODE='http', CHROMA_HOST='private-server', RAGDOC_LEXICAL_INDEX_PATH='/wrong/index')
    monkeypatch.setattr(os, 'environ', env)
    root = tmp_path / 'My papers – été'
    local.configure(root)
    assert env['RAGDOC_CHROMA_MODE'] == 'persistent'
    assert env['RAGDOC_MARKDOWN_DIR'] == str(root / 'articles_markdown')
    assert env['CHROMA_DB_PATH'] == str(root / 'chroma_db_new')
    assert env['PYTHON_DOTENV_DISABLED'] == '1'
    assert not {'VOYAGE_API_KEY', 'COHERE_API_KEY', 'CHROMA_HOST', 'RAGDOC_LEXICAL_INDEX_PATH'} & env.keys()


@pytest.mark.parametrize('sources', [[], ['../paper.md'], ['/paper.md'], ['paper.md;bad'], ['paper.pdf']])
def test_targeted_index_refuses_invalid_sources(sources):
    with pytest.raises(ValueError):
        local.validate_sources(sources)


def test_valid_targeted_source_and_missing_library(tmp_path):
    local.validate_sources(['paper_abcdef123456.md'])
    with pytest.raises(RuntimeError, match='Prepare this library'):
        local.collection(tmp_path)


def test_key_lookup_never_uses_shell_or_puts_value_in_arguments(monkeypatch):
    env = {}
    monkeypatch.setattr(os, 'environ', env)
    monkeypatch.setattr(local.sys, 'platform', 'darwin')
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='synthetic-key\n')
    monkeypatch.setattr(local.subprocess, 'run', run)
    local.load_voyage_key()
    assert env['VOYAGE_API_KEY'] == 'synthetic-key'
    assert calls[0][0][0] == '/usr/bin/security'
    assert 'synthetic-key' not in calls[0][0]
    assert not calls[0][1].get('shell')


def test_catalogue_detects_a_concurrent_revision_change(monkeypatch, tmp_path):
    import src.chroma_reads
    snapshots = iter([SimpleNamespace(metadata={'ragdoc_revision':'before'}),
                      SimpleNamespace(metadata={'ragdoc_revision':'after'})])
    monkeypatch.setattr(local, 'collection', lambda root: next(snapshots))
    monkeypatch.setattr(src.chroma_reads, 'read_collection', lambda *a, **k: {'metadatas':[]})
    with pytest.raises(RuntimeError, match='changed while reading'):
        local.documents(tmp_path)


def test_backup_preserves_managed_data_and_does_not_move_originals(tmp_path):
    files = {"articles_markdown/paper.md":"reviewed", "chroma_db_new/chroma.sqlite3":"fixture",
             "ragdoc_library/collection/snapshot.md":"canonical", "ragdoc_artifacts/table.json":"table"}
    for name, text in files.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    original = tmp_path/'original.pdf'
    original.write_bytes(b'original fixture')
    backup = local.backup_library(tmp_path)
    for name, text in files.items():
        assert (backup/name).read_text() == text
        assert (tmp_path/name).read_text() == text
    assert original.read_bytes() == b'original fixture'
    assert not (backup/'original.pdf').exists()


def test_repair_requires_explicit_confirmation_before_keys_or_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(local, 'load_voyage_key', lambda: pytest.fail('credential access before confirmation'))
    args = SimpleNamespace(root=tmp_path, action='repair', source=[], confirm_rebuild=False)
    with pytest.raises(ValueError, match='Confirm the rebuild'):
        local.dispatch(args)
    assert list(tmp_path.iterdir()) == []


def test_count_refuses_repairing_and_concurrent_revisions(monkeypatch, tmp_path):
    import src.chroma_reads
    args = SimpleNamespace(root=tmp_path, action='count', source=['paper.md'])
    monkeypatch.setattr(local, 'collection', lambda root: SimpleNamespace(metadata={'ragdoc_repairing':True}))
    with pytest.raises(RuntimeError, match='not ready'):
        local.dispatch(args)
    states = iter([SimpleNamespace(metadata={'ragdoc_revision':'a'}), SimpleNamespace(metadata={'ragdoc_revision':'b'})])
    monkeypatch.setattr(local, 'collection', lambda root: next(states))
    monkeypatch.setattr(src.chroma_reads, 'read_collection', lambda *a, **k: {'ids':['one']})
    with pytest.raises(RuntimeError, match='changed during verification'):
        local.dispatch(args)


def test_failed_backup_aborts_before_engine_initialization(monkeypatch, tmp_path):
    from scripts import index_incremental as indexer
    monkeypatch.setattr(indexer, 'VOYAGE_API_KEY', 'synthetic-key')
    monkeypatch.setattr(indexer, 'CHROMA_DB_PATH', tmp_path/'chroma')
    monkeypatch.setattr(indexer, 'LIBRARY_PATH', tmp_path/'canonical')
    monkeypatch.setattr(indexer.voyageai, 'Client', lambda **kw: pytest.fail('index engine reached after failed backup'))
    def fail_backup():
        assert (tmp_path/'.indexing.lock').exists()
        raise OSError('synthetic backup failure')
    with pytest.raises(OSError, match='synthetic backup failure'):
        indexer.index_incremental(force_reindex=True, before_write=fail_backup)
    assert not (tmp_path/'chroma').exists()
    # Failure released the existing writer lock.
    handle = indexer.acquire_lock(tmp_path/'.indexing.lock')
    assert handle is not None
    indexer.release_lock(tmp_path/'.indexing.lock', handle)

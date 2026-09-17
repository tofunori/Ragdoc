import json

from src.artifacts import ArtifactIndex


def test_indexes_and_finds_exact_table(tmp_path):
    folder = tmp_path / "ren"
    assets = folder / "assets"
    assets.mkdir(parents=True)
    (assets / "table.jpg").write_bytes(b"jpeg")
    (folder / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "source": "ren.md",
        "artifacts": [{
            "artifact_id": "table2",
            "type": "table",
            "label": "Table 2",
            "page": 7,
            "bbox": [1, 2, 3, 4],
            "caption": "Four BRDF parameterizations",
            "body": "P1 P2 P3 P4",
            "image": "assets/table.jpg",
        }],
    }), encoding="utf-8")
    index = ArtifactIndex(tmp_path)

    assert index.index() == {"sources": 1, "artifacts": 1}
    exact = index.search(source="ren.md", label="table 2")
    lexical = index.search(query="BRDF parameterizations")

    assert exact[0]["page"] == 7
    assert exact[0]["image_available"] is True
    assert lexical[0]["artifact_id"] == "table2"


def test_reindex_removes_deleted_artifact(tmp_path):
    folder = tmp_path / "paper"
    folder.mkdir()
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"source": "paper.md", "artifacts": [{
        "artifact_id": "figure1", "type": "image", "label": "Figure 1",
        "page": 2, "caption": "Map", "body": "", "image": None,
    }]}), encoding="utf-8")
    index = ArtifactIndex(tmp_path)
    index.index()
    manifest.write_text(json.dumps({"source": "paper.md", "artifacts": []}), encoding="utf-8")

    index.index({"paper.md"})

    assert index.search(source="paper.md") == []

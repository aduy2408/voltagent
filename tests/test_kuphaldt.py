from pathlib import Path

from app.rag.kuphaldt import KuphaldtParser, chunk_sections

HTML = """
<html><head><title>Lessons In Electric Circuits -- Volume I (DC) - Chapter 2</title></head>
<body>
<a href="DC_1.html"><img src="previous.jpg" alt="Previous"></a>
<h1>Lessons In Electric Circuits -- Volume I</h1>
<h1>Chapter 2</h1>
<h1>OHM'S LAW</h1>
<ul><li><a href="#one">Navigation entry</a></li></ul>
<h2>How voltage relates</h2>
<p>Voltage is potential energy per unit charge.</p>
<p><img src="10001.png" alt="voltage symbols"></p>
<table><tr><th>Quantity</th><th>Unit</th></tr><tr><td>Voltage</td><td>volt</td></tr></table>
<h3>Subtopic</h3>
<pre>V = I * R
.print dc v(1,0)</pre>
<h2>Contributors</h2><p>Names only.</p>
</body></html>
"""


def test_parser_preserves_structure_and_skips_site_chrome(tmp_path: Path):
    root = tmp_path / "kuphaldt"
    volume = root / "DC"
    volume.mkdir(parents=True)
    chapter = volume / "DC_2.html"
    chapter.write_text(HTML, encoding="utf-8")

    sections = KuphaldtParser(root).parse_file(chapter)

    assert [section.section_title for section in sections] == ["How voltage relates", "Subtopic"]
    assert "Previous" not in sections[0].content
    assert "Voltage is potential energy" in sections[0].content
    assert "Quantity | Unit" in sections[0].content
    assert "V = I * R" in sections[1].content
    assert sections[0].image_references == [str(volume / "10001.png")]
    assert sections[0].source_url.startswith("file:")


def test_discovery_ignores_navigation_and_appendix_files(tmp_path: Path):
    root = tmp_path / "kuphaldt"
    volume = root / "AC"
    volume.mkdir(parents=True)
    for name in ["AC_1.html", "AC_2.htm", "bye_1.html", "AC_A1.html", "index.html"]:
        (volume / name).write_text(HTML, encoding="utf-8")

    found = KuphaldtParser(root).discover_chapters(["AC"])

    assert [path.name for path in found] == ["AC_1.html", "AC_2.htm"]


def test_chunker_respects_size_overlap_and_stable_ids(tmp_path: Path):
    root = tmp_path / "kuphaldt"
    volume = root / "DC"
    volume.mkdir(parents=True)
    chapter = volume / "DC_2.html"
    chapter.write_text(HTML, encoding="utf-8")
    sections = KuphaldtParser(root).parse_file(chapter)

    first = chunk_sections(sections, chunk_size=8, chunk_overlap=2)
    second = chunk_sections(sections, chunk_size=8, chunk_overlap=2)

    assert first
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.token_count <= 8 for chunk in first)
    assert all(chunk.document_id == sections[chunk.parent_section_index].document_id for chunk in first)

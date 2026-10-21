from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_VOLUMES = ("DC", "AC", "Semi", "Digital", "Ref", "Exper")
_VOLUME_PREFIXES = {
    "DC": "DC",
    "AC": "AC",
    "Semi": "SEMI",
    "Digital": "DIGI",
    "Ref": "REF",
    "Exper": "EXP",
}
_VOLUME_NAMES = {
    "DC": "Direct Current",
    "AC": "Alternating Current",
    "Semi": "Semiconductors",
    "Digital": "Digital Circuits",
    "Ref": "Electrical Reference",
    "Exper": "Experiments",
}
_NAV_IMAGE_NAMES = {"previous.jpg", "contents.jpg", "next.jpg"}
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "pre", "table", "ul", "ol", "blockquote"}
_TOKEN_RE = re.compile(r"\S+")


class DocumentSection(BaseModel):
    """A meaningful, structure-aware section extracted from one chapter page."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    volume: str
    chapter_number: int
    chapter_title: str
    section_title: str
    content: str
    source_path: str
    source_url: str
    image_references: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A retrieval-ready chunk retaining its parent section provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    volume: str
    chapter_number: int
    chapter_title: str
    section_title: str
    content: str
    processed_content: str
    source_path: str
    source_url: str
    image_references: list[str] = Field(default_factory=list)
    token_count: int
    chunk_index: int
    parent_section_index: int

    @field_validator("token_count")
    @classmethod
    def positive_token_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("token_count must be positive")
        return value


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class ParseError:
    path: str
    error: str


@dataclass
class ParseReport:
    """Corpus inspection and parsing statistics."""

    eligible_chapter_files: int = 0
    parsed_chapters: int = 0
    parsed_sections: int = 0
    image_references: int = 0
    skipped_files: list[SkippedFile] = field(default_factory=list)
    errors: list[ParseError] = field(default_factory=list)


class KuphaldtParser:
    """Parse the downloaded Kuphaldt HTML corpus without indexing site chrome."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def discover_chapters(self, volumes: Iterable[str] | None = None) -> list[Path]:
        selected = tuple(volumes or SUPPORTED_VOLUMES)
        unknown = sorted(set(selected) - set(SUPPORTED_VOLUMES))
        if unknown:
            raise ValueError(f"Unsupported volume(s): {', '.join(unknown)}")
        files: list[Path] = []
        for volume in selected:
            directory = self.root / volume
            prefix = _VOLUME_PREFIXES[volume]
            if not directory.is_dir():
                continue
            pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.html?$", re.IGNORECASE)
            files.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and pattern.fullmatch(path.name)
            )
        return sorted(files, key=lambda path: (path.parent.name, _chapter_number(path.name), path.name))

    def parse_file(self, path: str | Path) -> list[DocumentSection]:
        file_path = Path(path).expanduser().resolve()
        volume = file_path.parent.name
        chapter_number = _chapter_number(file_path.name)
        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        body = soup.body or soup
        headings = body.find_all(["h1", "h2", "h3", "h4"])
        chapter_title = _chapter_title(soup, headings, volume, chapter_number)
        try:
            relative_path = file_path.relative_to(self.root).as_posix()
        except ValueError:
            relative_path = file_path.name
        document_id = _stable_id(volume, str(chapter_number), relative_path)

        sections: list[DocumentSection] = []
        current_title = chapter_title
        current_blocks: list[str] = []
        current_images: list[str] = []

        for element in _content_blocks(body):
            if element.name == "h1":
                continue
            if element.name in {"ul", "ol"} and _is_toc_list(element):
                continue
            if element.name in {"h2", "h3", "h4"}:
                if current_blocks:
                    section = self._make_section(
                        document_id,
                        volume,
                        chapter_number,
                        chapter_title,
                        current_title,
                        current_blocks,
                        current_images,
                        file_path,
                    )
                    if section is not None:
                        sections.append(section)
                current_title = _clean_text(element.get_text(" ", strip=True)) or current_title
                current_blocks = []
                current_images = []
                continue
            text, images = _render_block(element, file_path)
            if text:
                current_blocks.append(text)
            current_images.extend(images)

        if current_blocks:
            section = self._make_section(
                document_id,
                volume,
                chapter_number,
                chapter_title,
                current_title,
                current_blocks,
                current_images,
                file_path,
            )
            if section is not None:
                sections.append(section)
        return sections

    def parse(
        self, volumes: Iterable[str] | None = None
    ) -> tuple[list[DocumentSection], ParseReport]:
        report = ParseReport()
        sections: list[DocumentSection] = []
        candidates = self.discover_chapters(volumes)
        report.eligible_chapter_files = len(candidates)
        candidate_set = set(candidates)
        selected = tuple(volumes or SUPPORTED_VOLUMES)
        for volume in selected:
            directory = self.root / volume
            if not directory.is_dir():
                report.skipped_files.append(SkippedFile(str(directory), "volume directory missing"))
                continue
            for path in sorted(directory.glob("*.html")) + sorted(directory.glob("*.htm")):
                if path.resolve() not in candidate_set:
                    report.skipped_files.append(SkippedFile(str(path), "navigation or non-chapter HTML"))
        for path in candidates:
            try:
                parsed = self.parse_file(path)
                if not parsed:
                    report.skipped_files.append(SkippedFile(str(path), "no substantive sections"))
                    continue
                sections.extend(parsed)
                report.parsed_chapters += 1
                report.parsed_sections += len(parsed)
                report.image_references += sum(len(section.image_references) for section in parsed)
            except (OSError, UnicodeError, ValueError) as exc:
                report.errors.append(ParseError(str(path), str(exc)))
        return sections, report

    @staticmethod
    def _make_section(
        document_id: str,
        volume: str,
        chapter_number: int,
        chapter_title: str,
        section_title: str,
        blocks: list[str],
        images: list[str],
        source_path: Path,
    ) -> DocumentSection | None:
        if section_title.casefold().strip() in {"contributors", "copyright"}:
            return None
        content = "\n\n".join(block for block in blocks if block).strip()
        if not content or _is_navigation_only(content):
            return None
        unique_images = list(dict.fromkeys(images))
        return DocumentSection(
            document_id=document_id,
            volume=_VOLUME_NAMES.get(volume, volume),
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            section_title=section_title,
            content=content,
            source_path=str(source_path),
            source_url=source_path.as_uri(),
            image_references=unique_images,
        )


def chunk_sections(
    sections: Iterable[DocumentSection], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[DocumentChunk]:
    """Chunk sections by token-like words while keeping block boundaries whenever possible."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for section_index, section in enumerate(sections):
        blocks = [block.strip() for block in section.content.split("\n\n") if block.strip()]
        units = list(_split_units(blocks, chunk_size, chunk_overlap))
        for local_index, (content, token_count) in enumerate(units):
            chunk_index = len(chunks)
            chunk_id = _stable_id(
                section.document_id,
                str(section_index),
                str(local_index),
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=section.document_id,
                    volume=section.volume,
                    chapter_number=section.chapter_number,
                    chapter_title=section.chapter_title,
                    section_title=section.section_title,
                    content=content,
                    processed_content=content,
                    source_path=section.source_path,
                    source_url=section.source_url,
                    image_references=section.image_references,
                    token_count=token_count,
                    chunk_index=chunk_index,
                    parent_section_index=section_index,
                )
            )
    return chunks


def inspect_corpus(
    root: str | Path, volumes: Iterable[str] | None = None, chunk_size: int = 512, chunk_overlap: int = 64
) -> tuple[list[DocumentSection], list[DocumentChunk], ParseReport]:
    parser = KuphaldtParser(root)
    sections, report = parser.parse(volumes)
    return sections, chunk_sections(sections, chunk_size, chunk_overlap), report


def _chapter_number(name: str) -> int:
    match = re.search(r"_(\d+)\.html?$", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Not a numeric chapter filename: {name}")
    return int(match.group(1))


def _chapter_title(
    soup: BeautifulSoup, headings: list[Tag], volume: str, chapter_number: int
) -> str:
    for heading in headings:
        text = _clean_text(heading.get_text(" ", strip=True))
        if (
            text
            and not re.fullmatch(r"chapter\s+\d+", text, re.IGNORECASE)
            and "Lessons In Electric Circuits" not in text
        ):
            return text
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    match = re.search(r"\)\s*-\s*Chapter\s+\d+", title, re.IGNORECASE)
    if match:
        return title[: match.start()].rsplit("-", 1)[-1].strip()
    return f"{_VOLUME_NAMES.get(volume, volume)} Chapter {chapter_number}"


def _content_blocks(body: Tag) -> Iterator[Tag]:
    for element in body.find_all(list(_BLOCK_TAGS)):
        if any(isinstance(parent, Tag) and parent.name in _BLOCK_TAGS for parent in element.parents):
            continue
        yield element


def _render_block(element: Tag, source_path: Path) -> tuple[str, list[str]]:
    images: list[str] = []
    for image in element.find_all("img"):
        source = image.get("src")
        if not source:
            continue
        name = Path(unquote(str(source))).name.casefold()
        if name in _NAV_IMAGE_NAMES:
            continue
        resolved = (source_path.parent / unquote(str(source))).resolve()
        images.append(str(resolved))
        label = _clean_text(image.get("alt", ""))
        marker = f"[Image: {resolved}"
        if label:
            marker += f"; alt: {label}"
        marker += "]"
        image.replace_with(NavigableString(marker))
    if element.name == "table":
        rows = []
        for row in element.find_all("tr"):
            cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows), images
    if element.name in {"ul", "ol"}:
        items = []
        for item in element.find_all("li", recursive=False):
            text = _clean_text(item.get_text(" ", strip=True))
            if text:
                items.append(f"- {text}")
        return "\n".join(items), images
    if element.name == "pre":
        text = element.get_text("\n", strip=False).strip("\n")
        return f"```text\n{text}\n```" if text else "", images
    text = _clean_text(element.get_text(" ", strip=True))
    return text, images


def _split_units(
    blocks: list[str], chunk_size: int, chunk_overlap: int
) -> Iterator[tuple[str, int]]:
    current: list[str] = []
    count = 0
    for block in blocks:
        block_tokens = _TOKEN_RE.findall(block)
        if not block_tokens:
            continue
        if len(block_tokens) > chunk_size:
            if current:
                yield "\n\n".join(current), count
                current, count = [], 0
            for start in range(0, len(block_tokens), chunk_size - chunk_overlap):
                piece = block_tokens[start : start + chunk_size]
                if piece:
                    yield " ".join(piece), len(piece)
            continue
        if current and count + len(block_tokens) > chunk_size:
            content = "\n\n".join(current)
            yield content, count
            overlap_tokens = _TOKEN_RE.findall(content)[-chunk_overlap:]
            if len(overlap_tokens) + len(block_tokens) <= chunk_size:
                current = [" ".join(overlap_tokens)] if overlap_tokens else []
                count = len(overlap_tokens)
            else:
                current, count = [], 0
        current.append(block)
        count += len(block_tokens)
    if current:
        yield "\n\n".join(current), count


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_navigation_only(content: str) -> bool:
    normalized = content.casefold()
    return normalized in {"previous contents next", "previous contents", "contents next", "previous next"}


def _is_toc_list(element: Tag) -> bool:
    items = element.find_all("li", recursive=False)
    if not items:
        return False
    return all(item.find("a", href=lambda href: isinstance(href, str) and href.startswith("#")) for item in items)


def _stable_id(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

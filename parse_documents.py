from __future__ import annotations

import json
import re
import zlib
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


TEXT_EXTENSIONS = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".md",
    ".log",
}
PARSEABLE_ZIP_EXTENSIONS = TEXT_EXTENSIONS | {".hwpx", ".zip", ".pdf", ".hwp"}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _text_quality_score(value: str) -> float:
    if not value:
        return 0.0
    allowed = 0
    for char in value:
        if (
            char.isalnum()
            or char.isspace()
            or char in ".,;:!?()[]{}<>-_/\\'\"@#%&*+=~|"
            or "\u3131" <= char <= "\u318E"
            or "\uAC00" <= char <= "\uD7A3"
        ):
            allowed += 1
    return allowed / len(value)


def _is_reasonable_text(value: str, min_length: int = 20, min_score: float = 0.7) -> bool:
    cleaned = _clean_text(value)
    if len(cleaned) < min_length:
        return False
    return _text_quality_score(cleaned) >= min_score


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _decode_bytes_with_fallback(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _payload(parser: str, full_text: str, **extra: Any) -> dict[str, Any]:
    cleaned = _clean_text(full_text)
    payload: dict[str, Any] = {
        "parser": parser,
        "full_text": cleaned,
        "text_length": len(cleaned),
        "text_excerpt": cleaned[:4000],
    }
    payload.update(extra)
    return payload


def parse_text_document(path: Path) -> dict[str, Any]:
    text = _read_text_with_fallback(path)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            normalized = json.dumps(data, ensure_ascii=False, indent=2)
            return _payload("json", normalized)
        except json.JSONDecodeError:
            pass
    return _payload("text", text)


def parse_hwpx(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        preview_text = ""
        if "Preview/PrvText.txt" in names:
            preview_text = _decode_bytes_with_fallback(archive.read("Preview/PrvText.txt"))

        section_texts: list[str] = []
        for name in sorted(names):
            if not re.match(r"Contents/section\d+\.xml$", name):
                continue
            xml_text = _decode_bytes_with_fallback(archive.read(name))
            try:
                root = ElementTree.fromstring(xml_text)
                section_texts.append(" ".join(t.strip() for t in root.itertext() if t.strip()))
            except ElementTree.ParseError:
                continue

    preview_clean = _clean_text(preview_text)
    section_merged = _clean_text("\n".join(section_texts))
    merged_text = section_merged if len(section_merged) > len(preview_clean) else preview_clean
    return _payload("hwpx", merged_text, archive_entries=names[:100], archive_entry_count=len(names))


def _extract_pdf_literal_strings(data: bytes) -> str:
    results: list[str] = []
    current: list[str] = []
    depth = 0
    escape = False

    for byte in data:
        char = chr(byte)
        if depth == 0:
            if char == "(":
                depth = 1
                current = []
            continue

        if escape:
            current.append(char)
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == "(":
            depth += 1
            current.append(char)
            continue

        if char == ")":
            depth -= 1
            if depth == 0:
                text = "".join(current)
                if any(ch.isalnum() for ch in text):
                    results.append(text)
                current = []
            else:
                current.append(char)
            continue

        current.append(char)

    return "\n".join(results)


def parse_pdf(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, flags=re.S)
    chunks: list[str] = []

    for stream in streams:
        candidates = [stream]
        try:
            candidates.append(zlib.decompress(stream))
        except zlib.error:
            pass

        for candidate in candidates:
            text_objects = re.findall(rb"BT(.*?)ET", candidate, flags=re.S)
            for text_object in text_objects:
                text = _extract_pdf_literal_strings(text_object)
                cleaned = _clean_text(text)
                if _is_reasonable_text(cleaned):
                    chunks.append(cleaned)

    merged = "\n".join(dict.fromkeys(chunks))
    parser = "pdf" if _is_reasonable_text(merged, min_length=40, min_score=0.75) else "pdf_binary"
    if parser == "pdf_binary":
        merged = ""
    note = None if merged else "No extractable PDF text found without external PDF parser."
    return _payload(parser, merged, note=note)


def _extract_hwp_strings(raw: bytes) -> str:
    patterns = [
        rb"(?:[\x20-\x7E][\x00]){4,}",
        rb"(?:[\xAC-\xD7][\x00-\xFF]){4,}",
    ]
    texts: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, raw):
            text = _decode_bytes_with_fallback(match)
            cleaned = _clean_text(text)
            if len(cleaned) >= 8:
                texts.append(cleaned)
    return "\n".join(dict.fromkeys(texts))


def parse_hwp(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    extracted = _extract_hwp_strings(raw)
    parser = "hwp" if _is_reasonable_text(extracted, min_length=40, min_score=0.65) else "hwp_binary"
    if parser == "hwp_binary":
        extracted = ""
    note = None if extracted else "Legacy HWP binary parser is limited in this environment."
    return _payload(parser, extracted, note=note)


def parse_zip(path: Path) -> dict[str, Any]:
    extract_root = path.parent / f"{path.stem}_unzipped"
    parsed_children: list[dict[str, Any]] = []
    collected_texts: list[str] = []

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        extract_root.mkdir(parents=True, exist_ok=True)

        for name in names[:50]:
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in PARSEABLE_ZIP_EXTENSIONS:
                continue

            target_path = extract_root / Path(name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(archive.read(name))

            child = parse_document(target_path)
            child_excerpt = child.get("text_excerpt", "")
            child_full_text = child.get("full_text", "")
            if child_full_text:
                collected_texts.append(f"[{target_path.name}]\n{child_full_text}")
            parsed_children.append(
                {
                    "path": str(target_path),
                    "name": target_path.name,
                    "parser": child.get("parser"),
                    "text_length": child.get("text_length"),
                    "text_excerpt": child_excerpt,
                    "error": child.get("error"),
                }
            )

    return _payload(
        "zip",
        "\n\n".join(collected_texts),
        archive_entry_count=len(names),
        archive_entries=names[:100],
        parsed_children=parsed_children,
    )


def parse_document(path: str | Path) -> dict[str, Any]:
    document_path = Path(path)
    result: dict[str, Any] = {
        "path": str(document_path),
        "name": document_path.name,
        "suffix": document_path.suffix.lower(),
        "size_bytes": document_path.stat().st_size if document_path.exists() else None,
    }

    try:
        suffix = document_path.suffix.lower()
        if suffix == ".hwpx":
            result.update(parse_hwpx(document_path))
        elif suffix == ".zip":
            result.update(parse_zip(document_path))
        elif suffix == ".pdf":
            result.update(parse_pdf(document_path))
        elif suffix == ".hwp":
            result.update(parse_hwp(document_path))
        elif suffix in TEXT_EXTENSIONS:
            result.update(parse_text_document(document_path))
        else:
            result.update(_payload("binary", "", note=f"Unsupported parser for {suffix or 'no extension'}"))
    except Exception as exc:
        result.update(
            {
                "parser": "error",
                "full_text": "",
                "text_length": 0,
                "text_excerpt": "",
                "error": str(exc),
            }
        )

    return result

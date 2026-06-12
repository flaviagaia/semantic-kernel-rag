"""Chunking estrutural de Markdown.

Em vez de cortar o texto a cada X tokens, dividimos pelos headers (##).
Cada chunk preserva uma unidade semântica completa (seção do documento),
o que melhora bastante a relevância do retrieval.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawChunk:
    chunk_id: str
    text: str
    source: str
    section: str


def chunk_markdown(path: Path) -> list[RawChunk]:
    """Divide um arquivo Markdown em chunks por seção (header ##)."""
    content = path.read_text(encoding="utf-8")

    # Título principal do documento (header #)
    title_match = re.search(r"^# (.+)$", content, flags=re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else path.stem

    # Divide pelas seções de segundo nível
    parts = re.split(r"^## ", content, flags=re.MULTILINE)
    chunks: list[RawChunk] = []

    for i, part in enumerate(parts[1:], start=1):
        lines = part.strip().splitlines()
        section = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        chunks.append(
            RawChunk(
                chunk_id=f"{path.stem}-{i}",
                # Prefixar com título + seção dá contexto ao embedding
                text=f"{doc_title} — {section}: {body}",
                source=path.name,
                section=section,
            )
        )
    return chunks


def load_documents(docs_dir: Path) -> list[RawChunk]:
    """Carrega e chunka todos os .md de um diretório."""
    chunks: list[RawChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks

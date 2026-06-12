"""RAG documental com Semantic Kernel.

Pipeline: Markdown → chunking estrutural → vector store → busca → resposta com fontes.

O backend é um InMemoryCollection para facilitar a execução local,
mas a abstração do SK permite trocar por Azure AI Search, Qdrant,
Redis ou Postgres/pgvector sem reescrever o pipeline.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.in_memory import InMemoryCollection
from semantic_kernel.contents import ChatHistory
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

from ingest import load_documents

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small


# Modelo de dados tipado: o SK gera embeddings automaticamente
# para o campo "vector" no upsert e na busca.
@vectorstoremodel
@dataclass
class DocChunk:
    chunk_id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]
    source: Annotated[str, VectorStoreField("data")]
    section: Annotated[str, VectorStoreField("data")]
    embedding: Annotated[
        list[float] | None,
        VectorStoreField("vector", dimensions=EMBEDDING_DIMENSIONS),
    ] = None

    def __post_init__(self):
        # O texto do chunk é a fonte do embedding
        if self.embedding is None:
            self.embedding = self.text  # type: ignore[assignment]


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY não configurada. "
            "Copie .env.example para .env e adicione sua chave."
        )

    embedder = OpenAITextEmbedding(
        ai_model_id=os.getenv("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    chat = OpenAIChatCompletion(
        ai_model_id=os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 1. Ingestão: chunking estrutural por headers
    raw_chunks = load_documents(DOCS_DIR)
    print(f"📂 {len(raw_chunks)} chunks gerados de {DOCS_DIR}")

    records = [
        DocChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            source=c.source,
            section=c.section,
        )
        for c in raw_chunks
    ]

    async with InMemoryCollection(
        record_type=DocChunk,
        collection_name="politicas",
        embedding_generator=embedder,
    ) as collection:
        await collection.ensure_collection_exists()
        await collection.upsert(records)  # embeddings gerados aqui

        pergunta = "Em quanto tempo um pedido pode ser cancelado e como funciona o reembolso?"
        print(f"\n🧑 Pergunta: {pergunta}")

        # 2. Retrieval: busca vetorial top-k
        resultados = await collection.search(values=pergunta, top=3)
        contexto, fontes = [], []
        async for item in resultados.results:
            contexto.append(item.record.text)
            fontes.append(f"{item.record.source} → {item.record.section} (score {item.score:.3f})")

        # 3. Geração fundamentada no contexto recuperado
        history = ChatHistory()
        history.add_system_message(
            "Você é um assistente de políticas internas. Responda APENAS com base "
            "no contexto fornecido. Se a resposta não estiver no contexto, diga que não sabe.\n\n"
            "Contexto:\n" + "\n---\n".join(contexto)
        )
        history.add_user_message(pergunta)

        resposta = await chat.get_chat_message_content(
            chat_history=history,
            settings=OpenAIChatPromptExecutionSettings(),
        )
        print(f"\n🤖 Resposta: {resposta.content}")
        print("\n📌 Fontes recuperadas:")
        for fonte in fontes:
            print(f"   • {fonte}")


if __name__ == "__main__":
    asyncio.run(main())

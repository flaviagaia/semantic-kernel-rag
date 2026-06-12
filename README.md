# RAG with Semantic Kernel — Typed Vector Store & Structure-Aware Chunking

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python · Semantic Kernel ≥ 1.43 · OpenAI Embeddings (text-embedding-3-small, 1536d)

---

## 🇧🇷 Português

### Visão geral

Pipeline de RAG documental que explora a camada de dados vetoriais do [Semantic Kernel](https://github.com/microsoft/semantic-kernel):

- **Chunking estrutural** — Markdown é dividido por headers (`##`), não por janela fixa de tokens. Cada chunk é uma unidade semântica completa (seção), prefixada com `título do documento — seção:` para enriquecer o sinal do embedding.
- **Modelo de dados tipado** — `@vectorstoremodel` sobre uma `dataclass` declara o schema do índice: campo `key`, campos `data` (payload) e campo `vector` com dimensionalidade explícita. O schema do conhecimento fica versionável junto com o código.
- **Embeddings automáticos** — registrando um `embedding_generator` na collection, o SK vetoriza no `upsert` (indexação) e na `search` (query). Não há orquestração manual de embeddings.
- **Geração fundamentada** — o prompt restringe o modelo ao contexto recuperado e a resposta sai acompanhada das fontes com score de similaridade (cosseno).

### Arquitetura

```
data/docs/*.md
     │  chunking por headers (## → 1 chunk por seção)
     ▼
DocChunk (@vectorstoremodel: key | data | vector[1536])
     │  embeddings automáticos no upsert
     ▼
InMemoryCollection ◄── mesma interface: Azure AI Search, Qdrant,
     │                 Redis, Postgres/pgvector, Chroma...
     │  busca vetorial top-k (cosine)
     ▼
LLM com contexto restrito + citação de fontes com score
```

### Por que chunking estrutural

Corte por tamanho fixo quebra tabelas, listas e frases no meio, degradando o recall do retrieval. Dividir pela estrutura do documento produz chunks autocontidos cuja fronteira coincide com a fronteira semântica do conteúdo. Para Markdown, os headers já são essa fronteira; o mesmo princípio se aplica a seções de PDF e delimitadores de TXT.

### A abstração de backend é o ponto forte

`InMemoryCollection` aqui é deliberado: o contrato `VectorStoreCollection` (CRUD + search) é idêntico para todos os conectores. Migrar de protótipo local para Azure AI Search em produção é trocar a instância da collection, não reescrever ingestão, busca ou geração.

### Execução

```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY
python src/main.py
```

### Estrutura

```
data/docs/           # base de conhecimento (políticas internas de exemplo)
src/
├── ingest.py        # chunking estrutural de Markdown
└── main.py          # ingestão → indexação → retrieval → geração com fontes
```

---

## 🇺🇸 English

### Overview

Document RAG pipeline exploring the vector-data layer of [Semantic Kernel](https://github.com/microsoft/semantic-kernel):

- **Structure-aware chunking** — Markdown is split by headers (`##`), not by a fixed token window. Each chunk is a complete semantic unit (a section), prefixed with `document title — section:` to enrich the embedding signal.
- **Typed data model** — `@vectorstoremodel` on a `dataclass` declares the index schema: a `key` field, `data` (payload) fields and a `vector` field with explicit dimensionality. Your knowledge schema becomes versionable alongside the code.
- **Automatic embeddings** — by registering an `embedding_generator` on the collection, SK vectorizes on `upsert` (indexing) and on `search` (querying). No manual embedding orchestration.
- **Grounded generation** — the prompt constrains the model to the retrieved context, and the answer ships with its sources and cosine-similarity scores.

### Architecture

```
data/docs/*.md
     │  header-based chunking (## → 1 chunk per section)
     ▼
DocChunk (@vectorstoremodel: key | data | vector[1536])
     │  automatic embeddings on upsert
     ▼
InMemoryCollection ◄── same interface: Azure AI Search, Qdrant,
     │                 Redis, Postgres/pgvector, Chroma...
     │  top-k vector search (cosine)
     ▼
LLM with constrained context + source citation with scores
```

### Why structure-aware chunking

Fixed-size splitting cuts tables, lists and sentences mid-way, degrading retrieval recall. Splitting along document structure yields self-contained chunks whose boundaries coincide with the content's semantic boundaries. For Markdown, headers already are that boundary; the same principle extends to PDF sections and TXT delimiters.

### The backend abstraction is the real win

`InMemoryCollection` is deliberate: the `VectorStoreCollection` contract (CRUD + search) is identical across all connectors. Moving from a local prototype to Azure AI Search in production means swapping the collection instance, not rewriting ingestion, retrieval or generation.

### Running

```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY
python src/main.py
```

### Layout

```
data/docs/           # knowledge base (sample internal policies)
src/
├── ingest.py        # structure-aware Markdown chunking
└── main.py          # ingestion → indexing → retrieval → grounded generation
```

---

Part of my LinkedIn series on Semantic Kernel → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)

"""
Offline MSMARCO-XI Index Builder
Builds FAISS + BM25 indexes from ai4bharat/MSMARCO-XI.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset

# ---------------------------------------------------------
# Python path
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.models.data_models import ChunkMetadata

from app.chunking.fixed import FixedChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.semantic import SemanticChunker

from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever

from app.utils.logger import logger


DATASET_NAME = "ai4bharat/MSMARCO-XI"


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):

        parts = []

        for item in value:

            text = extract_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    if isinstance(value, dict):

        for key in [
            "passage_text",
            "text",
            "passage",
            "content",
        ]:

            if key in value:

                text = extract_text(value[key])

                if text:
                    return text

        parts = []

        for item in value.values():

            text = extract_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value).strip()


# =========================================================
# DATASET
# =========================================================

def extract_documents(
    split: str,
    limit: int,
) -> List[Dict[str, str]]:

    logger.info(
        f"Loading dataset: {DATASET_NAME}"
    )

    logger.info(
        f"Split={split}, limit={limit}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
    )

    logger.info(
        f"Dataset loaded: {len(dataset)} records"
    )

    documents = []

    max_records = min(
        limit,
        len(dataset),
    )

    for index in range(max_records):

        record = dataset[index]

        query = extract_text(
            record.get("query")
        )

        passages = extract_text(
            record.get("passages")
        )

        answers = extract_text(
            record.get("answers")
        )

        parts = []

        if query:
            parts.append(
                f"Question: {query}"
            )

        if passages:
            parts.append(
                f"Passage: {passages}"
            )

        if answers:
            parts.append(
                f"Answer: {answers}"
            )

        text = "\n".join(parts).strip()

        if not text:
            continue

        query_id = extract_text(
            record.get("query_id")
        )

        if not query_id:
            query_id = str(index)

        documents.append(
            {
                "document_id": f"msmarco_xi_{query_id}",
                "text": text,
            }
        )

    logger.info(
        f"Extracted {len(documents)} searchable documents"
    )

    return documents


# =========================================================
# BUILD INDEX
# =========================================================

def build_index(
    split: str = "train",
    sample_limit: int = 1000,
):

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING MSMARCO-XI INDEXING")
    logger.info("=" * 60)

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    index_dir = Path(
        settings.INDEX_PATH
    ).parent

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        f"Index directory: {index_dir}"
    )

    # -----------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------

    logger.info(
        "STEP 1: Loading MSMARCO-XI..."
    )

    docs = extract_documents(
        split=split,
        limit=sample_limit,
    )

    if not docs:

        raise RuntimeError(
            "No documents were extracted from MSMARCO-XI."
        )

    # -----------------------------------------------------
    # 2. Chunking
    # -----------------------------------------------------

    logger.info(
        "STEP 2: Creating chunks..."
    )

    fixed_chunker = FixedChunker(
        chunk_size=80,
        overlap=20,
    )

    sentence_chunker = SentenceChunker(
        target_words=50,
    )

    semantic_chunker = SemanticChunker(
        similarity_threshold=0.5,
    )

    all_chunks: List[ChunkMetadata] = []

    seen_texts = set()

    for doc in docs:

        doc_id = doc["document_id"]
        text = doc["text"]

        chunks = []

        # Fixed chunking
        try:

            chunks.extend(
                fixed_chunker.chunk_text(
                    doc_id,
                    text,
                )
            )

        except Exception as error:

            logger.warning(
                f"Fixed chunking failed for "
                f"{doc_id}: {error}"
            )

        # Sentence chunking
        try:

            chunks.extend(
                sentence_chunker.chunk_text(
                    doc_id,
                    text,
                )
            )

        except Exception as error:

            logger.warning(
                f"Sentence chunking failed for "
                f"{doc_id}: {error}"
            )

        # Semantic chunking
        try:

            chunks.extend(
                semantic_chunker.chunk_text(
                    doc_id,
                    text,
                )
            )

        except Exception as error:

            logger.warning(
                f"Semantic chunking failed for "
                f"{doc_id}: {error}"
            )

        # Deduplicate
        for chunk in chunks:

            normalized = " ".join(
                chunk.text.lower().split()
            )

            if not normalized:
                continue

            if normalized in seen_texts:
                continue

            seen_texts.add(normalized)

            all_chunks.append(chunk)

    logger.info(
        f"Generated {len(all_chunks)} unique chunks"
    )

    if not all_chunks:

        raise RuntimeError(
            "No chunks were generated."
        )

    # -----------------------------------------------------
    # 3. FAISS
    # -----------------------------------------------------

    logger.info(
        "STEP 3: Building FAISS dense index..."
    )

    dense_retriever = DenseRetriever(
        settings.EMBEDDING_MODEL
    )

    dense_retriever.build_index(
        all_chunks
    )

    dense_retriever.save_index(
        settings.INDEX_PATH
    )

    logger.info(
        f"FAISS index saved: "
        f"{settings.INDEX_PATH}"
    )

    # -----------------------------------------------------
    # 4. BM25
    # -----------------------------------------------------

    logger.info(
        "STEP 4: Building BM25 index..."
    )

    bm25_retriever = BM25Retriever()

    bm25_retriever.build_index(
        all_chunks
    )

    bm25_retriever.save_index(
        settings.BM25_PATH
    )

    logger.info(
        f"BM25 index saved: "
        f"{settings.BM25_PATH}"
    )

    # -----------------------------------------------------
    # 5. Metadata
    # -----------------------------------------------------

    logger.info(
        "STEP 5: Saving metadata..."
    )

    metadata = []

    for chunk in all_chunks:

        if hasattr(chunk, "model_dump"):

            metadata.append(
                chunk.model_dump()
            )

        elif hasattr(chunk, "dict"):

            metadata.append(
                chunk.dict()
            )

        else:

            metadata.append(
                chunk.__dict__
            )

    with open(
        settings.METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # 6. Configuration
    # -----------------------------------------------------

    config_stats = {

        "dataset": DATASET_NAME,

        "split": split,

        "total_documents": len(docs),

        "total_chunks": len(all_chunks),

        "strategies": [
            "fixed",
            "sentence",
            "semantic",
        ],

        "embedding_model":
            settings.EMBEDDING_MODEL,

        "built_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }

    with open(
        settings.CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_stats,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # 7. Verify
    # -----------------------------------------------------

    logger.info(
        "STEP 6: Verifying generated files..."
    )

    required_files = [
        settings.INDEX_PATH,
        settings.BM25_PATH,
        settings.METADATA_PATH,
        settings.CONFIG_PATH,
    ]

    for file_path in required_files:

        path = Path(file_path)

        if not path.exists():

            raise RuntimeError(
                f"Index file was not created: {file_path}"
            )

        logger.info(
            f"OK: {file_path}"
        )

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("MSMARCO-XI INDEXING COMPLETE")
    logger.info(
        f"Documents: {len(docs)}"
    )
    logger.info(
        f"Chunks: {len(all_chunks)}"
    )
    logger.info(
        f"Time: {elapsed:.2f} seconds"
    )
    logger.info(
        f"Index directory: {index_dir}"
    )
    logger.info("=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build MSMARCO-XI RAG indexes"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=1000,
        help="Number of records to index",
    )

    args = parser.parse_args()

    build_index(
        split=args.split,
        sample_limit=args.sample,
    )

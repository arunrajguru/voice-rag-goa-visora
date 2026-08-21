"""
Offline Indexing Pipeline for Voice RAG System

Downloads ai4bharat/MSMARCO-XI, extracts query/passage content,
creates chunks, and builds the Dense + BM25 indexes.
"""

import os
import sys
import json
import time
import argparse

from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset

# Add backend directory to Python path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from app.config import settings
from app.models.data_models import ChunkMetadata

from app.chunking.fixed import FixedChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.semantic import SemanticChunker

from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever

from app.utils.logger import logger


DATASET_NAME = "ai4bharat/MSMARCO-XI"


# ==========================================================
# DATASET EXTRACTION
# ==========================================================

def extract_text(value: Any) -> str:
    """
    Convert different MSMARCO-XI field formats into plain text.
    """

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

        # Common passage field names
        for key in [
            "passage_text",
            "text",
            "passage",
            "content"
        ]:

            if key in value:

                text = extract_text(
                    value[key]
                )

                if text:
                    return text

        # Fallback: extract all values
        parts = []

        for item in value.values():

            text = extract_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value).strip()


def extract_documents(
    split: str,
    limit: int
) -> List[Dict[str, str]]:

    logger.info(
        f"Loading {DATASET_NAME}"
    )

    logger.info(
        f"Split={split}, limit={limit}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=split
    )

    logger.info(
        f"Dataset loaded: {len(dataset)} records"
    )

    documents = []

    max_records = min(
        limit,
        len(dataset)
    )

    for index in range(max_records):

        record = dataset[index]

        query = extract_text(
            record.get("query")
        )

        passages = record.get(
            "passages"
        )

        passage_text = extract_text(
            passages
        )

        answers = extract_text(
            record.get("answers")
        )

        # --------------------------------------------------
        # Build searchable text
        # --------------------------------------------------

        parts = []

        if query:
            parts.append(
                f"Question: {query}"
            )

        if passage_text:
            parts.append(
                f"Passage: {passage_text}"
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
                "text": text
            }
        )

    logger.info(
        f"Extracted {len(documents)} searchable documents"
    )

    return documents


# ==========================================================
# BUILD INDEX
# ==========================================================

def build_index(
    split: str = "train",
    sample_limit: int = 5000
):

    start_time = time.time()

    logger.info("=" * 60)
    logger.info(
        "STARTING MSMARCO-XI INDEXING"
    )
    logger.info("=" * 60)

    # ------------------------------------------------------
    # Output directory
    # ------------------------------------------------------

    out_dir = Path(
        settings.INDEX_PATH
    ).parent

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # 1. Load actual dataset
    # ------------------------------------------------------

    logger.info(
        "STEP 1: Loading MSMARCO-XI..."
    )

    docs = extract_documents(
        split=split,
        limit=sample_limit
    )

    if not docs:

        raise RuntimeError(
            "No documents were extracted from MSMARCO-XI."
        )

    # ------------------------------------------------------
    # 2. Chunking
    # ------------------------------------------------------

    logger.info(
        "STEP 2: Creating chunks..."
    )

    fixed_chunker = FixedChunker(
        chunk_size=80,
        overlap=20
    )

    sentence_chunker = SentenceChunker(
        target_words=50
    )

    semantic_chunker = SemanticChunker(
        similarity_threshold=0.5
    )

    all_chunks: List[
        ChunkMetadata
    ] = []

    seen_texts = set()

    for doc in docs:

        doc_id = doc["document_id"]
        text = doc["text"]

        chunks = []

        # Fixed
        try:
            chunks.extend(
                fixed_chunker.chunk_text(
                    doc_id,
                    text
                )
            )
        except Exception as e:

            logger.warning(
                f"Fixed chunking failed "
                f"for {doc_id}: {e}"
            )

        # Sentence
        try:
            chunks.extend(
                sentence_chunker.chunk_text(
                    doc_id,
                    text
                )
            )
        except Exception as e:

            logger.warning(
                f"Sentence chunking failed "
                f"for {doc_id}: {e}"
            )

        # Semantic
        try:
            chunks.extend(
                semantic_chunker.chunk_text(
                    doc_id,
                    text
                )
            )
        except Exception as e:

            logger.warning(
                f"Semantic chunking failed "
                f"for {doc_id}: {e}"
            )

        # Deduplicate
        for chunk in chunks:

            clean_text = (
                " ".join(
                    chunk.text.lower().split()
                )
            )

            if not clean_text:
                continue

            if clean_text in seen_texts:
                continue

            seen_texts.add(
                clean_text
            )

            all_chunks.append(
                chunk
            )

    logger.info(
        f"Generated {len(all_chunks)} unique chunks"
    )

    if not all_chunks:

        raise RuntimeError(
            "No chunks were generated."
        )

    # ------------------------------------------------------
    # 3. Dense index
    # ------------------------------------------------------

    logger.info(
        "STEP 3: Building dense index..."
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
        f"Dense index saved to "
        f"{settings.INDEX_PATH}"
    )

    # ------------------------------------------------------
    # 4. BM25 index
    # ------------------------------------------------------

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
        f"BM25 index saved to "
        f"{settings.BM25_PATH}"
    )

    # ------------------------------------------------------
    # 5. Metadata
    # ------------------------------------------------------

    logger.info(
        "STEP 5: Saving metadata..."
    )

    metadata = []

    for chunk in all_chunks:

        if hasattr(
            chunk,
            "model_dump"
        ):

            metadata.append(
                chunk.model_dump()
            )

        else:

            metadata.append(
                chunk.dict()
            )

    with open(
        settings.METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # 6. Statistics
    # ------------------------------------------------------

    config_stats = {

        "dataset": DATASET_NAME,

        "split": split,

        "total_documents": len(docs),

        "total_chunks": len(
            all_chunks
        ),

        "strategies": [
            "fixed",
            "sentence",
            "semantic"
        ],

        "embedding_model":
            settings.EMBEDDING_MODEL,

        "built_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    with open(
        settings.CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config_stats,
            file,
            indent=2
        )

    elapsed = (
        time.time()
        - start_time
    )

    logger.info("=" * 60)

    logger.info(
        "MSMARCO-XI INDEXING COMPLETE"
    )

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
        f"Output: {out_dir}"
    )

    logger.info("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="train"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Number of MSMARCO-XI records to index"
    )

    args = parser.parse_args()

    build_index(
        split=args.split,
        sample_limit=args.sample
    )

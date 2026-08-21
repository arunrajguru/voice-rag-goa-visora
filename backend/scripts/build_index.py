"""
MSMARCO-XI Offline Index Builder
HH Goa 2026 - Voice RAG

Downloads ai4bharat/MSMARCO-XI from Hugging Face,
extracts passages, creates chunks, and builds
lightweight Dense + BM25 indexes.

Designed to run on Render.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset


# ==========================================================
# PATH SETUP
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ==========================================================
# APPLICATION IMPORTS
# ==========================================================

from app.config import settings

from app.models.data_models import ChunkMetadata

from app.chunking.fixed import FixedChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.semantic import SemanticChunker

from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever

from app.utils.logger import logger


# ==========================================================
# DATASET
# ==========================================================

DATASET_NAME = "ai4bharat/MSMARCO-XI"


# ==========================================================
# TEXT EXTRACTION
# ==========================================================

def extract_text(value: Any) -> str:
    """
    Recursively extract text from strings,
    lists and dictionaries.
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

        # Common passage keys
        preferred_keys = [
            "passage_text",
            "text",
            "passage",
            "content",
            "title",
            "answer",
        ]

        for key in preferred_keys:

            if key in value:

                text = extract_text(
                    value[key]
                )

                if text:
                    return text

        # Fallback
        parts = []

        for item in value.values():

            text = extract_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value).strip()


# ==========================================================
# EXTRACT DATASET DOCUMENTS
# ==========================================================

def extract_documents(
    split: str,
    limit: int
) -> List[Dict[str, str]]:

    logger.info("=" * 60)

    logger.info(
        f"Loading dataset: {DATASET_NAME}"
    )

    logger.info(
        f"Split: {split}"
    )

    logger.info(
        f"Limit: {limit}"
    )

    logger.info("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split=split
    )

    logger.info(
        f"Dataset loaded successfully."
    )

    logger.info(
        f"Total records available: {len(dataset)}"
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

        passages = extract_text(
            record.get("passages")
        )

        answers = extract_text(
            record.get("answers")
        )

        # --------------------------------------------------
        # Build searchable document
        # --------------------------------------------------

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

        text = "\n".join(
            parts
        ).strip()

        if not text:
            continue

        query_id = extract_text(
            record.get("query_id")
        )

        if not query_id:

            query_id = str(index)

        documents.append(
            {
                "document_id":
                    f"msmarco_xi_{query_id}",

                "text": text
            }
        )

        # Progress logging
        if (index + 1) % 500 == 0:

            logger.info(
                f"Processed "
                f"{index + 1}/{max_records} records"
            )

    logger.info(
        f"Extracted {len(documents)} documents."
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
        "STARTING MSMARCO-XI INDEX BUILD"
    )

    logger.info("=" * 60)

    # ======================================================
    # CREATE INDEX DIRECTORY
    # ======================================================

    index_dir = Path(
        settings.INDEX_PATH
    ).parent

    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        f"Index directory: {index_dir}"
    )

    # ======================================================
    # STEP 1 — DATASET
    # ======================================================

    logger.info(
        "STEP 1/5: Loading MSMARCO-XI..."
    )

    documents = extract_documents(
        split=split,
        limit=sample_limit
    )

    if not documents:

        raise RuntimeError(
            "No documents were extracted "
            "from MSMARCO-XI."
        )

    # ======================================================
    # STEP 2 — CHUNKING
    # ======================================================

    logger.info(
        "STEP 2/5: Creating chunks..."
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

    for index, document in enumerate(
        documents
    ):

        document_id = document[
            "document_id"
        ]

        text = document[
            "text"
        ]

        chunks = []

        # --------------------------------------------------
        # Fixed chunks
        # --------------------------------------------------

        try:

            chunks.extend(
                fixed_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Fixed chunking failed "
                f"for {document_id}: "
                f"{error}"
            )

        # --------------------------------------------------
        # Sentence chunks
        # --------------------------------------------------

        try:

            chunks.extend(
                sentence_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Sentence chunking failed "
                f"for {document_id}: "
                f"{error}"
            )

        # --------------------------------------------------
        # Semantic chunks
        # --------------------------------------------------

        try:

            chunks.extend(
                semantic_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Semantic chunking failed "
                f"for {document_id}: "
                f"{error}"
            )

        # --------------------------------------------------
        # Deduplicate
        # --------------------------------------------------

        for chunk in chunks:

            normalized = (
                " ".join(
                    chunk.text
                    .lower()
                    .split()
                )
            )

            if not normalized:
                continue

            if normalized in seen_texts:
                continue

            seen_texts.add(
                normalized
            )

            all_chunks.append(
                chunk
            )

        if (index + 1) % 500 == 0:

            logger.info(
                f"Chunked "
                f"{index + 1}/{len(documents)} "
                f"documents"
            )

    logger.info(
        f"Total unique chunks: "
        f"{len(all_chunks)}"
    )

    if not all_chunks:

        raise RuntimeError(
            "No chunks were generated."
        )

    # ======================================================
    # STEP 3 — DENSE INDEX
    # ======================================================

    logger.info(
        "STEP 3/5: Building dense index..."
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
        f"Dense index saved: "
        f"{settings.INDEX_PATH}"
    )

    # ======================================================
    # STEP 4 — BM25
    # ======================================================

    logger.info(
        "STEP 4/5: Building BM25 index..."
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

    # ======================================================
    # STEP 5 — METADATA
    # ======================================================

    logger.info(
        "STEP 5/5: Saving metadata..."
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

    # ======================================================
    # CONFIGURATION
    # ======================================================

    config = {

        "dataset":
            DATASET_NAME,

        "split":
            split,

        "records":
            len(documents),

        "chunks":
            len(all_chunks),

        "chunking":
            [
                "fixed",
                "sentence",
                "semantic"
            ],

        "embedding_model":
            settings.EMBEDDING_MODEL,

        "dense_index":
            settings.INDEX_PATH,

        "bm25_index":
            settings.BM25_PATH,

        "metadata":
            settings.METADATA_PATH,

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
            config,
            file,
            indent=2
        )

    # ======================================================
    # FINAL VERIFICATION
    # ======================================================

    logger.info("=" * 60)

    logger.info(
        "VERIFYING INDEX FILES..."
    )

    files = [
        settings.INDEX_PATH,
        settings.BM25_PATH,
        settings.METADATA_PATH,
        settings.CONFIG_PATH,
    ]

    for file_path in files:

        path = Path(file_path)

        logger.info(
            f"{path.name}: "
            f"{'OK' if path.exists() else 'MISSING'}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    logger.info("=" * 60)

    logger.info(
        "MSMARCO-XI INDEX BUILD COMPLETE"
    )

    logger.info(
        f"Documents: {len(documents)}"
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


# ==========================================================
# COMMAND LINE
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Build Voice RAG index from MSMARCO-XI"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Number of records to index"
    )

    args = parser.parse_args()

    build_index(
        split=args.split,
        sample_limit=args.sample
    )

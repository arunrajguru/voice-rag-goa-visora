"""
MSMARCO-XI English Index Builder
Builds a lightweight Dense + BM25 index for Render deployment.
"""

import sys
import json
import time
from pathlib import Path
from typing import Any, List, Dict

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

# English configuration
DATASET_CONFIG = "en"


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

                text = extract_text(
                    value[key]
                )

                if text:
                    return text

        parts = []

        for value_item in value.values():

            text = extract_text(
                value_item
            )

            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value).strip()


# =========================================================
# LOAD DATASET
# =========================================================

def load_documents(
    split: str,
    limit: int
) -> List[Dict[str, str]]:

    logger.info(
        f"Loading {DATASET_NAME}"
    )

    logger.info(
        f"Configuration: {DATASET_CONFIG}"
    )

    logger.info(
        f"Split: {split}"
    )

    logger.info(
        f"Limit: {limit}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=f"{split}[:{limit}]",
    )

    logger.info(
        f"Loaded {len(dataset)} records"
    )

    documents = []

    for index, record in enumerate(dataset):

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
                "document_id":
                    f"msmarco_xi_{query_id}",
                "text": text,
            }
        )

    logger.info(
        f"Extracted {len(documents)} documents"
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
    logger.info("STARTING MSMARCO-XI ENGLISH INDEX BUILD")
    logger.info("=" * 60)

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    index_dir = Path(
        settings.INDEX_PATH
    ).parent

    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 1. DATASET
    # -----------------------------------------------------

    logger.info(
        "STEP 1: Loading English MSMARCO-XI..."
    )

    documents = load_documents(
        split=split,
        limit=sample_limit,
    )

    if not documents:

        raise RuntimeError(
            "No documents extracted from MSMARCO-XI."
        )

    # -----------------------------------------------------
    # 2. CHUNKING
    # -----------------------------------------------------

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

    all_chunks = []

    seen_texts = set()

    for document in documents:

        document_id = document[
            "document_id"
        ]

        text = document["text"]

        chunk_sets = []

        # Fixed chunks
        try:

            chunk_sets.extend(
                fixed_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Fixed chunking failed: "
                f"{error}"
            )

        # Sentence chunks
        try:

            chunk_sets.extend(
                sentence_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Sentence chunking failed: "
                f"{error}"
            )

        # Semantic chunks
        try:

            chunk_sets.extend(
                semantic_chunker.chunk_text(
                    document_id,
                    text
                )
            )

        except Exception as error:

            logger.warning(
                f"Semantic chunking failed: "
                f"{error}"
            )

        # Deduplicate
        for chunk in chunk_sets:

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
    # 3. DENSE INDEX
    # -----------------------------------------------------

    logger.info(
        "STEP 3: Building dense index..."
    )

    dense = DenseRetriever(
        settings.EMBEDDING_MODEL
    )

    dense.build_index(
        all_chunks
    )

    dense.save_index(
        settings.INDEX_PATH
    )

    logger.info(
        f"Dense index saved: "
        f"{settings.INDEX_PATH}"
    )

    # -----------------------------------------------------
    # 4. BM25
    # -----------------------------------------------------

    logger.info(
        "STEP 4: Building BM25 index..."
    )

    bm25 = BM25Retriever()

    bm25.build_index(
        all_chunks
    )

    bm25.save_index(
        settings.BM25_PATH
    )

    logger.info(
        f"BM25 index saved: "
        f"{settings.BM25_PATH}"
    )

    # -----------------------------------------------------
    # 5. METADATA
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 6. CONFIG
    # -----------------------------------------------------

    config = {

        "dataset":
            DATASET_NAME,

        "configuration":
            DATASET_CONFIG,

        "split":
            split,

        "documents

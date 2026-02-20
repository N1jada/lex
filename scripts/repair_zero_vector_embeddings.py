#!/usr/bin/env python
"""
One-off script to repair zero-vector dense embeddings in Qdrant collections.

ROOT CAUSE:
During UKSI + UKPGA ingestion, ~150 legislation sections exceeded the OpenAI
text-embedding-3-large 8,192 token limit. The batch embedding function caught
the BadRequestError and stored [0.0] * 1024 as the dense vector. These points
exist in Qdrant with correct payloads and BM25 sparse vectors, but are invisible
to dense/hybrid semantic search.

FIX:
Re-embed using token-aware chunking: split into ~4000-token windows with
200-token overlap, embed each chunk, average and L2-normalise.

Only the dense vector is updated — sparse vectors and payloads are untouched.

Usage:
    # Dry run (default) — report zero-vector points
    USE_CLOUD_QDRANT=true uv run python scripts/repair_zero_vector_embeddings.py

    # Actually fix the data
    USE_CLOUD_QDRANT=true uv run python scripts/repair_zero_vector_embeddings.py --apply

    # Fix specific collection only
    USE_CLOUD_QDRANT=true uv run python scripts/repair_zero_vector_embeddings.py \
        --apply --collection legislation_section
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from qdrant_client.models import PointVectors

from lex.core.embeddings import generate_dense_embedding_with_retry
from lex.core.qdrant_client import get_qdrant_client
from lex.settings import (
    EXPLANATORY_NOTE_COLLECTION,
    LEGISLATION_SECTION_COLLECTION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Collections that may have zero-vector points
AFFECTED_COLLECTIONS = [
    LEGISLATION_SECTION_COLLECTION,
    EXPLANATORY_NOTE_COLLECTION,
]

# Batch size for update_vectors API calls
UPDATE_BATCH_SIZE = 50

# Scroll batch size
SCROLL_BATCH_SIZE = 250


def is_zero_vector(vector: list[float]) -> bool:
    """Check if a dense vector is all zeros."""
    return all(v == 0.0 for v in vector)


def reconstruct_embedding_text(payload: dict, collection_name: str) -> str:
    """Reconstruct the text that should be embedded from a point's payload.

    Matches the logic in LegislationSection.get_embedding_text() and
    ExplanatoryNote embedding text construction.
    """
    if collection_name == LEGISLATION_SECTION_COLLECTION:
        title = payload.get("title", "")
        text = payload.get("text", "")
        return f"{title}\n\n{text}"
    elif collection_name == EXPLANATORY_NOTE_COLLECTION:
        return payload.get("text", "")
    else:
        return payload.get("text", "")


def repair_collection(
    client,
    collection_name: str,
    apply: bool = False,
) -> dict:
    """
    Find and repair zero-vector dense embeddings in a collection.

    Returns stats about the operation.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing: {collection_name}")
    logger.info("=" * 60)

    info = client.get_collection(collection_name)
    total_points = info.points_count
    logger.info(f"Total documents: {total_points:,}")

    stats = {
        "collection": collection_name,
        "total_records": total_points,
        "records_checked": 0,
        "zero_vectors_found": 0,
        "records_repaired": 0,
        "batches_sent": 0,
        "errors": 0,
    }

    offset = None
    batch_num = 0
    pending_updates: list[PointVectors] = []
    start_time = time.time()

    while True:
        batch_num += 1

        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=["dense"],
        )

        if not results:
            break

        for point in results:
            stats["records_checked"] += 1

            dense_vector = point.vector.get("dense") if isinstance(point.vector, dict) else None
            if dense_vector is None:
                continue

            if not is_zero_vector(dense_vector):
                continue

            stats["zero_vectors_found"] += 1

            if apply:
                # Reconstruct text and re-embed with chunking
                text = reconstruct_embedding_text(point.payload, collection_name)
                if not text.strip():
                    logger.warning(f"  Point {point.id}: empty text, skipping")
                    continue

                try:
                    new_vector = generate_dense_embedding_with_retry(text)
                    pending_updates.append(
                        PointVectors(
                            id=point.id,
                            vector={"dense": new_vector},
                        )
                    )
                    logger.info(f"  Embedded point {point.id} ({len(text)} chars, vector norm OK)")
                except Exception as e:
                    logger.error(f"  Failed to embed point {point.id}: {e}")
                    stats["errors"] += 1
                    continue

                # Send batch when we hit the limit
                if len(pending_updates) >= UPDATE_BATCH_SIZE:
                    try:
                        client.update_vectors(
                            collection_name=collection_name,
                            points=pending_updates,
                            wait=False,
                        )
                        stats["records_repaired"] += len(pending_updates)
                        stats["batches_sent"] += 1
                        logger.info(
                            f"  Sent batch {stats['batches_sent']}: "
                            f"{stats['records_repaired']} repaired so far"
                        )
                    except Exception as e:
                        logger.error(f"  Batch update failed: {e}")
                        stats["errors"] += len(pending_updates)
                    pending_updates = []

        # Progress logging
        if batch_num % 50 == 0:
            pct = 100 * stats["records_checked"] / total_points if total_points else 0
            elapsed = time.time() - start_time
            rate = stats["records_checked"] / elapsed if elapsed > 0 else 0
            eta = (total_points - stats["records_checked"]) / rate if rate > 0 else 0
            logger.info(
                f"  Progress: {stats['records_checked']:,} / {total_points:,} ({pct:.1f}%) "
                f"- Found {stats['zero_vectors_found']} zero vectors - ETA: {eta / 60:.1f}min"
            )

        offset = next_offset
        if offset is None:
            break

    # Send any remaining updates
    if pending_updates and apply:
        try:
            client.update_vectors(
                collection_name=collection_name,
                points=pending_updates,
                wait=True,  # Wait for final batch
            )
            stats["records_repaired"] += len(pending_updates)
            stats["batches_sent"] += 1
            logger.info(f"  Sent final batch {stats['batches_sent']}")
        except Exception as e:
            logger.error(f"  Final batch update failed: {e}")
            stats["errors"] += len(pending_updates)

    # Final summary
    elapsed = time.time() - start_time
    mode = "APPLIED" if apply else "DRY RUN"
    logger.info(f"\n  [{mode}] Results for {collection_name}:")
    logger.info(f"    Time:         {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"    Checked:      {stats['records_checked']:,}")
    logger.info(f"    Zero vectors: {stats['zero_vectors_found']}")
    if apply:
        logger.info(f"    Repaired:     {stats['records_repaired']}")
        logger.info(f"    Batches:      {stats['batches_sent']}")
        logger.info(f"    Errors:       {stats['errors']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Repair zero-vector dense embeddings in Qdrant collections"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply fixes (default is dry run)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=AFFECTED_COLLECTIONS,
        help="Fix only a specific collection",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ZERO-VECTOR EMBEDDING REPAIR")
    logger.info("=" * 60)
    logger.info(f"Mode: {'APPLY FIXES' if args.apply else 'DRY RUN (preview only)'}")
    logger.info("")
    logger.info("Plan:")
    logger.info("  1. Scroll through each collection with dense vectors")
    logger.info("  2. Identify points with all-zero dense vectors")
    logger.info("  3. Reconstruct embedding text from payload")
    logger.info("  4. Re-embed with token-aware chunking")
    logger.info("  5. Update ONLY the dense vector via update_vectors()")
    logger.info("     (sparse vectors and payloads are untouched)")
    logger.info("")

    if not args.apply:
        logger.info("Running in DRY RUN mode. Use --apply to fix data.\n")

    client = get_qdrant_client()

    collections = [args.collection] if args.collection else AFFECTED_COLLECTIONS

    all_stats = []
    total_start = time.time()

    for collection_name in collections:
        try:
            stats = repair_collection(
                client,
                collection_name,
                apply=args.apply,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {collection_name}: {e}")
            all_stats.append(
                {
                    "collection": collection_name,
                    "error": str(e),
                }
            )

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_zero = sum(s.get("zero_vectors_found", 0) for s in all_stats)
    total_repaired = sum(s.get("records_repaired", 0) for s in all_stats)
    total_errors = sum(s.get("errors", 0) for s in all_stats)

    for stats in all_stats:
        if "error" in stats:
            logger.info(f"  {stats['collection']}: ERROR - {stats['error']}")
        else:
            status = "REPAIRED" if args.apply and stats["records_repaired"] > 0 else "FOUND"
            logger.info(
                f"  {stats['collection']}: {stats['zero_vectors_found']} zero vectors [{status}]"
            )

    logger.info(f"\n  Total time:         {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    logger.info(f"  Total zero vectors: {total_zero}")
    if args.apply:
        logger.info(f"  Total repaired:     {total_repaired}")
        logger.info(f"  Total errors:       {total_errors}")


if __name__ == "__main__":
    main()

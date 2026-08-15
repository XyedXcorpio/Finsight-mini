"""
pipelines/ingest_sec.py
========================

Weekend 1: download SEC 10-Q filings and land them in the Bronze Delta table.

What this does:
1. Downloads N most recent 10-Q filings per ticker via SEC EDGAR.
2. Extracts plain text from each filing's primary HTML document.
3. Writes one row per filing to s3a://finsight-bronze/sec_filings, with
   an `ingestion_timestamp` column (matches the PA1 pattern).

Bronze principle: minimal transformation. We extract text from HTML because
storing raw HTML bytes in a Spark string column is impractical, but we do
NOT clean whitespace, strip boilerplate, or extract sections here — that's
Silver's job. Bronze = "what we received," Silver = "what's usable."

Usage:
    python -m pipelines.ingest_sec --tickers NVDA AMD INTC TSLA AAPL --quarters 2

Requires SEC_EDGAR_CONTACT_EMAIL in .env — SEC EDGAR requires a real contact
email in the User-Agent header for all automated requests (fair-access policy,
not an API key). See: https://www.sec.gov/os/webmaster-faq#developers
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    TimestampType,
)

sys.path.insert(0, ".")
from common.spark_utils import get_spark  # noqa: E402

load_dotenv()

CONTACT_EMAIL = os.environ.get("SEC_EDGAR_CONTACT_EMAIL")
DOWNLOAD_DIR = Path("data/_sec_raw")  # scratch dir, not committed (gitignored)
BRONZE_PATH = "s3a://finsight-bronze/sec_filings"

BRONZE_SCHEMA = StructType([
    StructField("ticker", StringType(), nullable=False),
    StructField("filing_type", StringType(), nullable=False),
    StructField("accession_number", StringType(), nullable=False),
    StructField("primary_doc_filename", StringType(), nullable=True),
    StructField("raw_text", StringType(), nullable=False),
    StructField("source", StringType(), nullable=False),
    StructField("ingestion_timestamp", TimestampType(), nullable=False),
])


def _extract_text_from_html(html_path: Path) -> str:
    """
    Pull plain text out of a filing's primary HTML document.

    SEC filings are messy HTML — heavy on nested tables, inline styles, and
    XBRL tags. BeautifulSoup's get_text() with a separator handles this well
    enough for Bronze; deeper cleanup (whitespace collapsing, boilerplate
    removal) happens in Silver.
    """
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n")


def _find_primary_doc(accession_dir: Path) -> Path | None:
    """
    Locate the primary filing document inside a downloaded accession folder.

    sec-edgar-downloader saves each filing under:
        data/_sec_raw/sec-edgar-filings/{ticker}/{form}/{accession}/
    The primary document is almost always the largest .htm/.html file
    (exhibits and attachments are smaller).
    """
    html_files = list(accession_dir.glob("*.htm")) + list(accession_dir.glob("*.html"))
    if not html_files:
        return None
    return max(html_files, key=lambda p: p.stat().st_size)


def download_filings(tickers: list[str], quarters: int) -> list[dict]:
    """
    Download `quarters` most recent 10-Q filings for each ticker and
    extract their text. Returns a list of dicts ready for Spark.
    """
    if not CONTACT_EMAIL:
        raise RuntimeError(
            "SEC_EDGAR_CONTACT_EMAIL not set in .env. SEC EDGAR requires a "
            "real contact email in the User-Agent header for all automated "
            "requests — see https://www.sec.gov/os/webmaster-faq#developers"
        )

    from sec_edgar_downloader import Downloader

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader("FinSightMini", CONTACT_EMAIL, str(DOWNLOAD_DIR))

    rows: list[dict] = []
    now = datetime.now(timezone.utc)

    for ticker in tickers:
        logger.info(f"Downloading {quarters} most recent 10-Q filings for {ticker}...")
        try:
            dl.get("10-Q", ticker, limit=quarters, download_details=True)
        except Exception as e:
            logger.error(f"  Failed to download filings for {ticker}: {e}")
            continue

        ticker_dir = DOWNLOAD_DIR / "sec-edgar-filings" / ticker / "10-Q"
        if not ticker_dir.exists():
            logger.warning(f"  No filings directory found for {ticker}, skipping")
            continue

        accession_dirs = sorted(ticker_dir.iterdir())
        for accession_dir in accession_dirs:
            if not accession_dir.is_dir():
                continue

            primary_doc = _find_primary_doc(accession_dir)
            if primary_doc is None:
                logger.warning(f"  No HTML document found in {accession_dir.name}, skipping")
                continue

            try:
                text = _extract_text_from_html(primary_doc)
            except Exception as e:
                logger.error(f"  Failed to parse {primary_doc}: {e}")
                continue

            if len(text.strip()) < 500:
                logger.warning(
                    f"  Extracted text suspiciously short ({len(text)} chars) "
                    f"for {accession_dir.name}, skipping"
                )
                continue

            rows.append({
                "ticker": ticker,
                "filing_type": "10-Q",
                "accession_number": accession_dir.name,
                "primary_doc_filename": primary_doc.name,
                "raw_text": text,
                "source": "sec_edgar",
                "ingestion_timestamp": now,
            })
            logger.info(f"  {accession_dir.name}: extracted {len(text):,} chars")

        time.sleep(1)

    return rows


def write_to_bronze(rows: list[dict]) -> None:
    if not rows:
        logger.error("No rows to write — all downloads/extractions failed.")
        sys.exit(1)

    spark = get_spark("ingest_sec")
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)

    logger.info(f"Writing {df.count()} filings to {BRONZE_PATH}")
    df.write.format("delta").mode("append").save(BRONZE_PATH)

    logger.info("Bronze write complete. Verifying...")
    result = spark.read.format("delta").load(BRONZE_PATH)
    logger.info(f"Bronze table now has {result.count()} total rows")
    result.select("ticker", "filing_type", "accession_number", "ingestion_timestamp").show(
        truncate=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest SEC 10-Q filings into Bronze")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument(
        "--quarters", type=int, default=2, help="Number of recent 10-Qs per ticker"
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded raw files in data/_sec_raw (default: delete after ingest)",
    )
    args = parser.parse_args()

    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Quarters per ticker: {args.quarters}")

    rows = download_filings(args.tickers, args.quarters)
    write_to_bronze(rows)

    if not args.keep_raw and DOWNLOAD_DIR.exists():
        logger.info(f"Cleaning up {DOWNLOAD_DIR} (pass --keep-raw to preserve)")
        shutil.rmtree(DOWNLOAD_DIR)

    logger.info("Done. Next: python -m pipelines.silver_transform")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
pipelines/silver_transform.py
===============================

Weekend 1: clean Bronze SEC filings into Silver.

What this does:
1. Reads raw filing text from Bronze.
2. Normalizes whitespace (SEC HTML-to-text extraction leaves lots of
   irregular blank lines and spacing).
3. Attempts to isolate the "Risk Factors" section (Item 1A) — this is the
   section your later RAG queries will care about most, and it's a well-
   defined, consistently-labeled section in every 10-Q.
4. Computes word_count as a sanity check that cleaning didn't destroy
   real content.
5. Writes to s3a://finsight-silver/sec_filings_clean.

Usage:
    python -m pipelines.silver_transform
"""

from __future__ import annotations

import re
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

sys.path.insert(0, ".")
from common.spark_utils import get_spark  # noqa: E402

BRONZE_PATH = "s3a://finsight-bronze/sec_filings"
SILVER_PATH = "s3a://finsight-silver/sec_filings_clean"

# 10-Q filings consistently label sections as "Item 1A. Risk Factors" and the
# next section as "Item 2." (Management's Discussion and Analysis, or
# Unregistered Sales of Equity Securities depending on filing structure).
# This regex is deliberately permissive about whitespace/case since the
# HTML-to-text extraction can introduce irregular spacing around headers.
RISK_FACTORS_PATTERN = re.compile(
    r"item\s+1a\.?\s+risk\s+factors(.*?)item\s+2\.",
    re.IGNORECASE | re.DOTALL,
)


def clean_whitespace(text: str) -> str:
    """Collapse repeated blank lines/spaces left over from HTML extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_risk_factors(text: str) -> str | None:
    """
    Best-effort extraction of the Risk Factors section.

    SEC filings list every section in a Table of Contents near the top,
    e.g. "Item 1A. Risk Factors ... 33" (just a page number). A naive
    first-match regex grabs this TOC line instead of the real section,
    which appears later in the document. We search ALL occurrences of the
    pattern and keep the longest one — the TOC entry is always short (just
    headers and a page number), so the real section reliably wins once we
    stop stopping at the first match.

    Returns None if no occurrence has substantial content — this is
    expected for filings that incorporate risk factors "by reference" to
    their prior 10-K instead of restating them (common when there have
    been no material changes quarter-to-quarter).
    """
    matches = RISK_FACTORS_PATTERN.finditer(text)
    candidates = [m.group(1).strip() for m in matches]
    if not candidates:
        return None

    longest = max(candidates, key=len)
    # Sanity floor: a real Risk Factors section is at least a few hundred
    # characters. The TOC entry (just headers + a page number) will be far
    # shorter and won't survive this even as the "longest" candidate when
    # no real section exists.
    if len(longest) < 300:
        return None
    return longest


def main() -> int:
    spark = get_spark("silver_transform")

    print(f">>> Reading Bronze: {BRONZE_PATH}")
    bronze_df = spark.read.format("delta").load(BRONZE_PATH)
    bronze_count = bronze_df.count()
    print(f"    {bronze_count} filings in Bronze")

    clean_udf = F.udf(clean_whitespace, StringType())
    risk_udf = F.udf(extract_risk_factors, StringType())

    silver_df = (
        bronze_df
        .withColumn("clean_text", clean_udf(F.col("raw_text")))
        .withColumn("risk_factors_text", risk_udf(F.col("clean_text")))
        .withColumn("word_count", F.size(F.split(F.col("clean_text"), r"\s+")))
        .withColumn(
            "has_risk_factors_section",
            F.col("risk_factors_text").isNotNull(),
        )
        .drop("raw_text")  # Bronze keeps the raw copy; no need to duplicate it
    )

    print(">>> Writing Silver")
    silver_df.write.format("delta").mode("overwrite").save(SILVER_PATH)

    print(">>> Verifying Silver")
    result = spark.read.format("delta").load(SILVER_PATH)
    result.select(
        "ticker", "accession_number", "word_count", "has_risk_factors_section"
    ).orderBy("ticker", "accession_number").show(truncate=False)

    with_risk = result.filter(F.col("has_risk_factors_section")).count()
    print(f"    {with_risk}/{result.count()} filings had an extractable Risk Factors section")

    avg_words = result.agg(F.avg("word_count")).collect()[0][0]
    print(f"    average word_count: {avg_words:.0f}")

    if avg_words < 1000:
        print(
            "    WARNING: average word count looks low for a 10-Q — "
            "check that HTML extraction in ingest_sec.py grabbed the right document"
        )

    print()
    print(">>> Silver transform complete.")
    print("    Next: Weekend 2 — semantic chunking + embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
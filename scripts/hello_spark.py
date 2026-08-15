"""
scripts/hello_spark.py
=======================

Smoke test for Phase 0 infrastructure (FinSight Mini).

Verifies:
1. Spark starts with Delta + S3A (MinIO) configured.
2. We can write a Delta table to MinIO.
3. We can read it back with the correct schema.
4. Time travel works (proves the Delta transaction log is real).

Run:
    python scripts/hello_spark.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

sys.path.insert(0, ".")
from common.spark_utils import get_spark  # noqa: E402

# Mini only uses Bronze + Silver — no Gold layer.
LAYERS = {
    "bronze": "s3a://finsight-bronze/_smoke_test",
    "silver": "s3a://finsight-silver/_smoke_test",
}


def main() -> int:
    spark = get_spark("hello_spark")

    print(">>> Spark started")
    print(f"    version: {spark.version}")
    print()

    schema = StructType([
        StructField("ticker", StringType(), nullable=False),
        StructField("source", StringType(), nullable=False),
        StructField("ingestion_timestamp", TimestampType(), nullable=False),
    ])

    now = datetime.now(timezone.utc)
    rows = [
        ("NVDA", "smoke_test", now),
        ("AMD", "smoke_test", now),
        ("INTC", "smoke_test", now),
        ("TSLA", "smoke_test", now),
        ("AAPL", "smoke_test", now),
    ]
    df = spark.createDataFrame(rows, schema=schema)

    for layer, path in LAYERS.items():
        print(f">>> Writing to {layer}: {path}")
        (
            df.withColumn("layer", F.lit(layer))
              .write.format("delta").mode("overwrite").save(path)
        )

    print(">>> Reading back from bronze")
    bronze_df = spark.read.format("delta").load(LAYERS["bronze"])
    bronze_df.show(truncate=False)

    count = bronze_df.count()
    assert count == 5, f"Expected 5 rows, got {count}"
    print(f"    row count: {count} (ok)")

    print(">>> Demonstrating Delta time travel")
    extra = spark.createDataFrame(
        [("GOOGL", "smoke_test_v2", datetime.now(timezone.utc))],
        schema=schema,
    ).withColumn("layer", F.lit("bronze"))
    extra.write.format("delta").mode("append").save(LAYERS["bronze"])

    current_count = spark.read.format("delta").load(LAYERS["bronze"]).count()
    v0_count = (
        spark.read.format("delta")
             .option("versionAsOf", 0)
             .load(LAYERS["bronze"])
             .count()
    )
    print(f"    current version count: {current_count}")
    print(f"    version 0 count:       {v0_count}")
    assert current_count == 6 and v0_count == 5, "Time travel broken"
    print("    time travel works (ok)")

    print()
    print(">>> Phase 0 smoke test PASSED")
    print("    You are ready to start Weekend 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

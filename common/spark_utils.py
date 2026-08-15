"""
common.spark_utils
===================

Single source of truth for a SparkSession configured for Delta Lake + MinIO.

Usage:
    from common.spark_utils import get_spark
    spark = get_spark("ingest_sec")
    df = spark.read.format("delta").load("s3a://finsight-bronze/sec_filings")
"""

from __future__ import annotations

import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

_DEFAULTS = {
    "MINIO_ENDPOINT": "http://localhost:9000",
    "MINIO_ACCESS_KEY": "minioadmin",
    "MINIO_SECRET_KEY": "minioadmin",
    "SPARK_DRIVER_MEMORY": "4g",
    "SPARK_EXECUTOR_MEMORY": "4g",
    "SPARK_SHUFFLE_PARTITIONS": "8",
}


def _cfg(key: str) -> str:
    return os.environ.get(key, _DEFAULTS[key])


def get_spark(app_name: str = "finsight-mini") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.endpoint", _cfg("MINIO_ENDPOINT"))
        .config("spark.hadoop.fs.s3a.access.key", _cfg("MINIO_ACCESS_KEY"))
        .config("spark.hadoop.fs.s3a.secret.key", _cfg("MINIO_SECRET_KEY"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.committer.name", "magic")
        .config("spark.driver.memory", _cfg("SPARK_DRIVER_MEMORY"))
        .config("spark.executor.memory", _cfg("SPARK_EXECUTOR_MEMORY"))
        .config("spark.sql.shuffle.partitions", _cfg("SPARK_SHUFFLE_PARTITIONS"))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=[
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ],
    ).getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark() -> None:
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()

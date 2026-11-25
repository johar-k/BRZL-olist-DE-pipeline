
from pathlib import Path

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable


# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_PATH = (
    PROJECT_ROOT / "data/delta/orders"
)


# SPARK SESSION WITH DELTA LAKE

builder = (
    SparkSession.builder
    .appName("Stage8_Delta_Lake")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
)

spark = configure_spark_with_delta_pip(
    builder
).getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


try:

    # STEP 1: CREATE INITIAL TARGET TABLE IF NECESSARY

    if not DeltaTable.isDeltaTable(
        spark, str(TARGET_PATH)
    ):

        print("\n========== CREATING INITIAL DELTA TABLE ==========")

        initial_data = [
            (101, "C01", "Pending"),
            (102, "C02", "Shipped"),
            (103, "C03", "Delivered"),
        ]

        initial_df = spark.createDataFrame(
            initial_data,
            ["order_id", "customer_id", "status"]
        )

        (
            initial_df.write
            .format("delta")
            .mode("overwrite")
            .save(str(TARGET_PATH))
        )

    # STEP 2: LOAD DELTA TABLE

    target = DeltaTable.forPath(
        spark,
        str(TARGET_PATH)
    )

    print("\n========== ORIGINAL TARGET ==========")

    target.toDF().show()

    # STEP 3: CREATE INCOMING CDC EVENTS

    changes_data = [
        (101, "C01", "Shipped", "UPDATE"),
        (104, "C04", "Pending", "INSERT"),
        (103, "C03", None, "DELETE"),
    ]

    changes = spark.createDataFrame(
        changes_data,
        [
            "order_id",
            "customer_id",
            "status",
            "operation"
        ]
    )

    print("\n========== INCOMING CDC EVENTS ==========")

    changes.show()

    # STEP 4: APPLY CDC USING DELTA MERGE

    (
        target.alias("target")
        .merge(
            changes.alias("source"),
            "target.order_id = source.order_id"
        )
        .whenMatchedDelete(
            condition="source.operation = 'DELETE'"
        )
        .whenMatchedUpdate(
            condition="source.operation = 'UPDATE'",
            set={
                "status": "source.status"
            }
        )
        .whenNotMatchedInsert(
            condition="source.operation = 'INSERT'",
            values={
                "order_id": "source.order_id",
                "customer_id": "source.customer_id",
                "status": "source.status"
            }
        )
        .execute()
    )

    # STEP 5: DISPLAY UPDATED TABLE

    print("\n========== UPDATED DELTA TABLE ==========")

    target.toDF().orderBy("order_id").show()

    # STEP 6: DISPLAY TRANSACTION HISTORY

    print("\n========== DELTA HISTORY ==========")

    target.history().select(
        "version",
        "timestamp",
        "operation"
    ).show(truncate=False)

    # STEP 7: READ PREVIOUS TABLE VERSION

    print("\n========== ORIGINAL TABLE VERSION 0 ==========")

    original_version = (
        spark.read
        .format("delta")
        .option("versionAsOf", 0)
        .load(str(TARGET_PATH))
    )

    original_version.orderBy("order_id").show()

    # STEP 8: VERIFY DELTA TABLE

    print("\n========== DELTA TABLE INFORMATION ==========")

    print(
        "Is Delta table:",
        DeltaTable.isDeltaTable(
            spark,
            str(TARGET_PATH)
        )
    )

finally:
    spark.stop()

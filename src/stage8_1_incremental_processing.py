
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATH = (
    PROJECT_ROOT
    / "data/silver/olist_orders_dataset"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/incremental/orders"
)

STATE_DIR = (
    PROJECT_ROOT
    / "data/incremental/state"
)

WATERMARK_FILE = STATE_DIR / "orders_watermark.txt"

# Change this value for each simulated batch.
BATCH_END = "2017-03-31 23:59:59"

INITIAL_WATERMARK = "1900-01-01 00:00:00"


# --------------------------------------------------
# SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Stage8_Incremental_Processing")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    # STEP 1: CREATE STATE DIRECTORY

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 2: READ PREVIOUS WATERMARK

    if WATERMARK_FILE.exists():
        last_watermark = WATERMARK_FILE.read_text().strip()
    else:
        last_watermark = INITIAL_WATERMARK

    print("\n========== WATERMARK ==========")
    print("Previous watermark:", last_watermark)
    print("Current batch end:", BATCH_END)

    # STEP 3: READ SOURCE DATA

    orders = spark.read.parquet(str(SOURCE_PATH))

    # STEP 4: FILTER INCREMENTAL RECORDS

    incremental_orders = orders.filter(
        (col("order_purchase_timestamp") > lit(last_watermark).cast("timestamp"))
        &
        (col("order_purchase_timestamp") <= lit(BATCH_END).cast("timestamp"))
    )

    # STEP 5: VALIDATE BATCH

    batch_count = incremental_orders.count()

    print("\n========== BATCH INFORMATION ==========")
    print("New records:", batch_count)

    if batch_count > 0:

        # STEP 6: WRITE INCREMENTAL DATA

        (
            incremental_orders
            .write
            .mode("append")
            .parquet(str(OUTPUT_PATH))
        )

        print("Incremental data written successfully.")

    else:
        print("No new records found in this batch.")

    # STEP 7: UPDATE WATERMARK

    WATERMARK_FILE.write_text(BATCH_END)

    print("\n========== PROCESSING COMPLETED ==========")
    print("New watermark:", BATCH_END)

finally:
    spark.stop()

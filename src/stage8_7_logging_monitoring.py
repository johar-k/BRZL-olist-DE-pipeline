
import logging
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# 1. Create log directory
log_directory = Path("output/logs")
log_directory.mkdir(parents=True, exist_ok=True)


# 2. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            log_directory / "pipeline.log"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("orders_pipeline")


# 3. Create Spark session
spark = (
    SparkSession.builder
    .appName("Stage8_7_LoggingMonitoring")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def run_pipeline():

    start_time = time.time()

    try:
        logger.info("Orders pipeline started")

        # 4. Read Silver orders
        input_path = "data/silver/olist_orders_dataset"

        logger.info("Reading Silver orders dataset")

        orders = spark.read.parquet(input_path)

        # 5. Record input count
        input_count = orders.count()

        logger.info(
            "Input records received: %s",
            input_count
        )

        # 6. Validate records
        invalid_orders = orders.filter(
            F.col("order_id").isNull()
        )

        invalid_count = invalid_orders.count()

        logger.info(
            "Invalid records found: %s",
            invalid_count
        )

        # 7. Clean valid records
        valid_orders = (
            orders
            .filter(F.col("order_id").isNotNull())
            .dropDuplicates(["order_id"])
        )

        output_count = valid_orders.count()

        logger.info(
            "Valid records after cleaning: %s",
            output_count
        )

        # 8. Save output
        output_path = "output/stage8_7/orders"

        valid_orders.write \
            .mode("overwrite") \
            .parquet(output_path)

        logger.info(
            "Output successfully written to %s",
            output_path
        )

        # 9. Record execution metrics
        duration = round(time.time() - start_time, 2)

        logger.info("Pipeline execution completed")
        logger.info("Input records: %s", input_count)
        logger.info("Invalid records: %s", invalid_count)
        logger.info("Output records: %s", output_count)
        logger.info("Execution duration: %s seconds", duration)

    except Exception:
        logger.exception("Orders pipeline failed")
        raise

    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    run_pipeline()

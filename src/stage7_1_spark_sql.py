
from pyspark.sql import SparkSession

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("Stage7_Spark_SQL")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. Read Silver orders data
orders_df = spark.read.parquet(
    "data/silver/olist_orders_dataset"
)

# 3. Register temporary SQL view
orders_df.createOrReplaceTempView("orders")

# 4. Display available columns
print("\n========== ORDERS SCHEMA ==========")
orders_df.printSchema()

# 5. Total orders by status
print("\n========== ORDERS BY STATUS ==========")

status_result = spark.sql("""
    SELECT
        order_status,
        COUNT(*) AS total_orders
    FROM orders
    GROUP BY order_status
    ORDER BY total_orders DESC
""")

status_result.show(truncate=False)

# 6. Orders by purchase year
print("\n========== ORDERS BY YEAR ==========")

year_result = spark.sql("""
    SELECT
        YEAR(order_purchase_timestamp) AS purchase_year,
        COUNT(*) AS total_orders
    FROM orders
    WHERE order_purchase_timestamp IS NOT NULL
    GROUP BY YEAR(order_purchase_timestamp)
    ORDER BY purchase_year
""")

year_result.show()

# 7. Compare SQL and DataFrame APIs
print("\n========== DATAFRAME API RESULT ==========")

from pyspark.sql.functions import count

dataframe_result = (
    orders_df
    .groupBy("order_status")
    .agg(count("*").alias("total_orders"))
    .orderBy("total_orders", ascending=False)
)

dataframe_result.show(truncate=False)

# 8. Inspect the SQL execution plan
print("\n========== EXECUTION PLAN ==========")

status_result.explain(mode="formatted")

spark.stop()


from pyspark.sql import functions as F


def clean_orders(df):
    """
    Remove records without order_id,
    remove duplicate order IDs,
    and retain valid order records.
    """

    cleaned = (
        df.filter(F.col("order_id").isNotNull())
        .dropDuplicates(["order_id"])
        .filter(F.col("amount") >= 0)
    )

    return cleaned


import json
import time
from pathlib import Path
from datetime import datetime, timezone

input_path = Path("data/streaming/input")
input_path.mkdir(parents=True, exist_ok=True)

orders = [
    ("ORD101", "CUS01", 250.00, "PLACED"),
    ("ORD102", "CUS02", 450.50, "PLACED"),
    ("ORD103", "CUS03", 120.00, "PAID"),
    ("ORD104", "CUS04", 890.00, "PLACED"),
    ("ORD105", "CUS05", 300.00, "SHIPPED"),
    ("ORD106", "CUS06", 150.00, "PAID"),
]

for order_id, customer_id, amount, status in orders:

    event = {
        "order_id": order_id,
        "customer_id": customer_id,
        "amount": amount,
        "status": status,
        "event_time": datetime.now(timezone.utc).isoformat()
    }

    filename = f"{order_id}.json"
    final_path = input_path / filename
    temporary_path = input_path / f"{filename}.tmp"

    # Write a complete file before making it visible to Spark.
    temporary_path.write_text(json.dumps(event), encoding="utf-8")
    temporary_path.rename(final_path)

    print(f"Produced event: {order_id}")

    time.sleep(3)

print("Finished generating order events.")

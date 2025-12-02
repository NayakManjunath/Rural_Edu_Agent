# observability/obs.py
"""
Simple logging + basic metrics. Extend with Prometheus exporters or a small dashboard.
"""
import logging
import csv
from pathlib import Path
DATA_DIR = Path("data")
METRICS_FILE = DATA_DIR / "metrics.csv"
if not METRICS_FILE.exists():
    METRICS_FILE.write_text("timestamp,event,value\n")

logger = logging.getLogger("rural_agent")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def record_metric(event: str, value: float=1.0, timestamp=None):
    import time
    ts = timestamp or time.time()
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, event, value])
    logger.info("metric recorded: %s=%s", event, value)

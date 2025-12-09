import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

SPLIT_RATIOS = (0.8, 0.1, 0.1)

DB_URI = os.environ.get("DATABASE_URL", "mysql://user:pass@localhost/planpunk")

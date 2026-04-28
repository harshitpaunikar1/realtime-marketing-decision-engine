"""
ETL pipeline for real-time marketing decision engine.
Ingests feedback from surveys, chat, social, and CRM; normalizes and enriches before storage.
"""
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeedbackChannel(str, Enum):
    SURVEY = "survey"
    CHAT = "chat"
    SOCIAL = "social"
    CRM = "crm"
    EMAIL = "email"
    REVIEW = "review"


@dataclass
class RawFeedback:
    source_id: str
    channel: FeedbackChannel
    raw_text: str
    customer_id: Optional[str]
    campaign_id: Optional[str]
    received_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedFeedback:
    feedback_id: str
    source_id: str
    channel: str
    clean_text: str
    customer_id: Optional[str]
    campaign_id: Optional[str]
    received_at: float
    processed_at: float
    word_count: int
    contains_complaint: bool
    contains_praise: bool
    urgency_score: float
    language_detected: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextCleaner:
    """Normalizes raw feedback text."""

    COMPLAINT_PATTERNS = [
        r"\b(terrible|awful|broken|failed|refund|cancel|worst|horrible|useless|disappointed)\b",
        r"\b(not working|doesn't work|stopped working|issue|problem|bug|crash)\b",
    ]
    PRAISE_PATTERNS = [
        r"\b(great|excellent|amazing|love|fantastic|perfect|brilliant|outstanding)\b",
        r"\b(helpful|recommend|thank you|appreciate|impressed|satisfied)\b",
    ]

    def clean(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def detect_complaint(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.COMPLAINT_PATTERNS)

    def detect_praise(self, text: str) -> bool:
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.PRAISE_PATTERNS)

    def urgency_score(self, text: str) -> float:
        urgency_words = ["urgent", "immediately", "asap", "emergency", "critical",
                         "broken", "cannot access", "not working", "blocked"]
        text_lower = text.lower()
        hits = sum(1 for w in urgency_words if w in text_lower)
        return min(hits / 3.0, 1.0)

    def detect_language(self, text: str) -> str:
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        return "en"


class FeedbackNormalizer:
    """Transforms raw feedback records into enriched form."""

    def __init__(self):
        self.cleaner = TextCleaner()

    def normalize(self, raw: RawFeedback) -> EnrichedFeedback:
        clean = self.cleaner.clean(raw.raw_text)
        feedback_id = hashlib.md5(
            f"{raw.source_id}{raw.received_at}".encode()
        ).hexdigest()[:16]
        return EnrichedFeedback(
            feedback_id=feedback_id,
            source_id=raw.source_id,
            channel=raw.channel.value,
            clean_text=clean,
            customer_id=raw.customer_id,
            campaign_id=raw.campaign_id,
            received_at=raw.received_at,
            processed_at=time.time(),
            word_count=len(clean.split()),
            contains_complaint=self.cleaner.detect_complaint(clean),
            contains_praise=self.cleaner.detect_praise(clean),
            urgency_score=self.cleaner.urgency_score(clean),
            language_detected=self.cleaner.detect_language(clean),
            metadata=raw.metadata,
        )

    def normalize_batch(self, records: List[RawFeedback]) -> List[EnrichedFeedback]:
        return [self.normalize(r) for r in records]


class MarketingDB:
    """SQLite persistence for the marketing decision engine."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS enriched_feedback (
        feedback_id TEXT PRIMARY KEY,
        source_id TEXT,
        channel TEXT,
        clean_text TEXT,
        customer_id TEXT,
        campaign_id TEXT,
        received_at REAL,
        processed_at REAL,
        word_count INTEGER,
        contains_complaint INTEGER,
        contains_praise INTEGER,
        urgency_score REAL,
        language_detected TEXT,
        metadata TEXT
    );
    CREATE TABLE IF NOT EXISTS campaign_events (
        event_id TEXT PRIMARY KEY,
        campaign_id TEXT,
        event_type TEXT,
        channel TEXT,
        customer_id TEXT,
        metric_name TEXT,
        metric_value REAL,
        event_ts REAL
    );
    CREATE TABLE IF NOT EXISTS etl_log (
        run_id TEXT,
        started_at REAL,
        records_in INTEGER,
        records_out INTEGER,
        errors INTEGER,
        duration_ms REAL
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def insert_feedback(self, records: List[EnrichedFeedback]) -> int:
        rows = []
        for r in records:
            rows.append((
                r.feedback_id, r.source_id, r.channel, r.clean_text,
                r.customer_id, r.campaign_id, r.received_at, r.processed_at,
                r.word_count, int(r.contains_complaint), int(r.contains_praise),
                r.urgency_score, r.language_detected, json.dumps(r.metadata),
            ))
        self.conn.executemany(
            """INSERT OR IGNORE INTO enriched_feedback VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def insert_campaign_event(self, event_id: str, campaign_id: str, event_type: str,
                               channel: str, customer_id: Optional[str],
                               metric_name: str, metric_value: float,
                               event_ts: float) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO campaign_events VALUES (?,?,?,?,?,?,?,?)""",
            (event_id, campaign_id, event_type, channel, customer_id,
             metric_name, metric_value, event_ts),
        )
        self.conn.commit()

    def query_df(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn)

    def log_run(self, run_id: str, started_at: float, records_in: int,
                records_out: int, errors: int, duration_ms: float) -> None:
        self.conn.execute(
            "INSERT INTO etl_log VALUES (?,?,?,?,?,?)",
            (run_id, started_at, records_in, records_out, errors, duration_ms),
        )
        self.conn.commit()


class MarketingETL:
    """
    Orchestrates ingestion, normalization, enrichment, and storage of marketing feedback.
    """

    def __init__(self, db: MarketingDB):
        self.db = db
        self.normalizer = FeedbackNormalizer()

    def run(self, raw_records: List[RawFeedback]) -> Dict[str, Any]:
        run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        t0 = time.perf_counter()
        errors = 0
        enriched = []
        for r in raw_records:
            try:
                enriched.append(self.normalizer.normalize(r))
            except Exception as exc:
                logger.error("Normalization failed for %s: %s", r.source_id, exc)
                errors += 1

        stored = self.db.insert_feedback(enriched)
        duration_ms = (time.perf_counter() - t0) * 1000
        self.db.log_run(run_id, t0, len(raw_records), stored, errors, duration_ms)
        logger.info(
            "ETL run %s: in=%d enriched=%d stored=%d errors=%d %.0fms",
            run_id, len(raw_records), len(enriched), stored, errors, duration_ms,
        )
        return {
            "run_id": run_id,
            "records_in": len(raw_records),
            "records_enriched": len(enriched),
            "stored": stored,
            "errors": errors,
            "duration_ms": round(duration_ms, 1),
        }

    def complaint_rate(self, campaign_id: Optional[str] = None, hours: float = 24.0) -> float:
        since = time.time() - hours * 3600
        where = f"received_at >= {since}"
        if campaign_id:
            where += f" AND campaign_id = '{campaign_id}'"
        df = self.db.query_df(
            f"SELECT contains_complaint FROM enriched_feedback WHERE {where}"
        )
        if df.empty:
            return 0.0
        return round(float(df["contains_complaint"].mean()), 4)

    def channel_volume(self, hours: float = 24.0) -> pd.DataFrame:
        since = time.time() - hours * 3600
        return self.db.query_df(f"""
            SELECT channel,
                   COUNT(*) AS total,
                   SUM(contains_complaint) AS complaints,
                   SUM(contains_praise) AS praises,
                   ROUND(AVG(urgency_score), 3) AS avg_urgency
            FROM enriched_feedback
            WHERE received_at >= {since}
            GROUP BY channel
            ORDER BY total DESC
        """)


def _generate_demo_data(n: int = 200) -> List[RawFeedback]:
    rng = np.random.default_rng(42)
    channels = list(FeedbackChannel)
    texts = [
        "The product is absolutely amazing, highly recommend it!",
        "This is terrible, the app keeps crashing and I cannot access my account.",
        "Delivery was on time and packaging was great, thank you!",
        "URGENT: order not received, please refund immediately.",
        "Great experience overall, customer support was very helpful.",
        "The quality has dropped significantly, very disappointed.",
        "Works perfectly, exactly what I needed for my business.",
        "Cannot log in at all, broken since yesterday, need help asap.",
        "Amazing service, will definitely buy again next month.",
        "Product stopped working after one week, this is unacceptable.",
    ]
    campaigns = ["camp_001", "camp_002", "camp_003", None]
    records = []
    for i in range(n):
        records.append(RawFeedback(
            source_id=f"src_{i:04d}",
            channel=channels[rng.integers(0, len(channels))],
            raw_text=texts[rng.integers(0, len(texts))],
            customer_id=f"cust_{rng.integers(1000, 9999)}",
            campaign_id=campaigns[rng.integers(0, len(campaigns))],
            received_at=time.time() - rng.integers(0, 86400),
        ))
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = MarketingDB()
    etl = MarketingETL(db=db)
    raw = _generate_demo_data(200)

    result = etl.run(raw)
    print("ETL run result:", result)

    print("\nChannel volume (last 24h):")
    print(etl.channel_volume(24).to_string(index=False))

    print(f"\nOverall complaint rate: {etl.complaint_rate():.1%}")
    print(f"Campaign camp_001 complaint rate: {etl.complaint_rate('camp_001'):.1%}")

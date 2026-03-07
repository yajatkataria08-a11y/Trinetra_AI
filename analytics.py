import os
import logging

from config import BASE_DIR
from database import DatabaseConnection

logger = logging.getLogger("Trinetra")


class AnalyticsTracker:

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "analytics.db")
        self.db = DatabaseConnection(db_path)
        self._create_tables()

    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT, modality TEXT, results_count INTEGER,
                timestamp TEXT, search_duration_ms REAL, user TEXT
            )
        """)

    def log_search(self, query: str, modality: str, results_count: int,
                   duration_ms: float, user: str = "unknown"):
        try:
            self.db.execute("""
                INSERT INTO searches
                (query_text, modality, results_count, timestamp, search_duration_ms, user)
                VALUES (?, ?, ?, datetime('now'), ?, ?)
            """, (query, modality, results_count, duration_ms, user))
            logger.info(
                f"SEARCH user={user} modality={modality} "
                f"q={query!r} hits={results_count} ms={duration_ms:.0f}"
            )
        except Exception as e:
            logger.error(f"Failed to log search: {e}", exc_info=True)

    def get_stats(self, days: int = 7):
        days = max(1, min(int(days), 365))
        return self.db.fetchone(f"""
            SELECT COUNT(*), AVG(results_count), AVG(search_duration_ms)
            FROM searches
            WHERE timestamp >= datetime('now', '-{days} days')
        """)

    def get_popular_searches(self, limit: int = 10):
        return self.db.fetchall("""
            SELECT query_text, COUNT(*) as freq
            FROM searches
            WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY query_text ORDER BY freq DESC LIMIT ?
        """, (limit,))

    def get_search_suggestions(self, query: str, limit: int = 5) -> list:
        if not query or len(query) < 2:
            popular = self.get_popular_searches(limit)
            return [p[0] for p in popular]
        result = self.db.fetchall("""
            SELECT DISTINCT query_text, COUNT(*) as freq
            FROM searches WHERE query_text LIKE ? AND query_text != ?
            GROUP BY query_text ORDER BY freq DESC LIMIT ?
        """, (f"%{query}%", query, limit))
        return [row[0] for row in result]

import os
import json
import sqlite3
import logging
import threading

from config import BASE_DIR

logger = logging.getLogger("Trinetra")


# ==================== THREAD-SAFE DATABASE CONNECTION ====================

class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local  = threading.local()

    def get_connection(self):
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def execute(self, query, params=None):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params) if params else cursor.execute(query)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            conn.rollback()
            logger.error(f"DB execute error: {e}", exc_info=True)
            raise e

    def fetchall(self, query, params=None):
        return self.execute(query, params).fetchall()

    def fetchone(self, query, params=None):
        return self.execute(query, params).fetchone()

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# ==================== METADATA DB ====================

class MetadataDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "metadata.db")
        self.db = DatabaseConnection(db_path)
        self._create_tables()

    def _create_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, modality TEXT NOT NULL, language TEXT NOT NULL,
                file_path TEXT NOT NULL, file_size INTEGER, upload_date TEXT NOT NULL,
                faiss_index INTEGER NOT NULL, tags TEXT, description TEXT, collection TEXT,
                quality_score REAL, uploaded_by TEXT, UNIQUE(id)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL, user TEXT NOT NULL,
                comment TEXT NOT NULL, timestamp TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                asset_id TEXT NOT NULL, user TEXT NOT NULL,
                rating INTEGER NOT NULL, PRIMARY KEY (asset_id, user),
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
        """)

    def add_asset(self, asset_id, modality, language, file_path, file_size,
                  faiss_idx, tags=None, description="", collection="",
                  quality_score=None, uploaded_by="unknown"):
        try:
            self.db.execute("""
                INSERT INTO assets
                (id, modality, language, file_path, file_size, upload_date,
                 faiss_index, tags, description, collection, quality_score, uploaded_by)
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)
            """, (asset_id, modality, language, file_path, file_size,
                  faiss_idx, json.dumps(tags or []), description, collection,
                  quality_score, uploaded_by))
            logger.info(f"ASSET_ADDED id={asset_id} modality={modality} by={uploaded_by}")
        except sqlite3.IntegrityError:
            pass

    def search_metadata(self, modality=None, language=None, tags=None,
                        date_from=None, collection=None):
        query  = "SELECT id, faiss_index FROM assets WHERE 1=1"
        params = []
        if modality:   query += " AND modality=?";     params.append(modality)
        if language:   query += " AND language=?";     params.append(language)
        if date_from:  query += " AND upload_date>=?"; params.append(date_from)
        if tags:       query += " AND tags LIKE ?";    params.append(f"%{tags}%")
        if collection: query += " AND collection=?";   params.append(collection)
        return self.db.fetchall(query, params)

    def get_all_tags(self):
        result   = self.db.fetchall("SELECT DISTINCT tags FROM assets WHERE tags != '[]'")
        all_tags = set()
        for row in result:
            all_tags.update(json.loads(row[0]))
        return sorted(list(all_tags))

    def get_all_collections(self):
        result = self.db.fetchall(
            "SELECT DISTINCT collection FROM assets WHERE collection != ''"
        )
        return [row[0] for row in result]

    def add_comment(self, asset_id, user, comment):
        self.db.execute("""
            INSERT INTO comments (asset_id, user, comment, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (asset_id, user, comment))

    def get_comments(self, asset_id):
        return self.db.fetchall("""
            SELECT user, comment, timestamp FROM comments
            WHERE asset_id=? ORDER BY timestamp DESC
        """, (asset_id,))

    def add_rating(self, asset_id, user, rating):
        self.db.execute(
            "INSERT OR REPLACE INTO ratings (asset_id, user, rating) VALUES (?, ?, ?)",
            (asset_id, user, rating),
        )

    def get_rating(self, asset_id):
        result = self.db.fetchone(
            "SELECT AVG(rating), COUNT(*) FROM ratings WHERE asset_id=?", (asset_id,)
        )
        return result if result[0] else (0, 0)


# ==================== SINGLETON ====================

_shared_metadata_db = None

def get_shared_metadata_db() -> MetadataDB:
    global _shared_metadata_db
    if _shared_metadata_db is None:
        _shared_metadata_db = MetadataDB()
    return _shared_metadata_db

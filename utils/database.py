import sqlite3

DB_NAME = "health.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            risk TEXT,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_prediction(disease, risk, score):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (disease, risk, score) VALUES (?, ?, ?)",
        (disease, risk, score)
    )
    conn.commit()
    conn.close()

def fetch_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

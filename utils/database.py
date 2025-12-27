import sqlite3
import bcrypt
from datetime import datetime

DB_PATH = "health.db"

# =====================================================
# DB CONNECTION
# =====================================================
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# =====================================================
# CREATE TABLES
# =====================================================
def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash BLOB,
            role TEXT,
            created_at TEXT
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            condition TEXT,
            result TEXT,
            confidence REAL,
            summary TEXT,
            created_at TEXT
        )
    """)

    # Contact messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            subject TEXT,
            message TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

# =====================================================
# USER MANAGEMENT
# =====================================================
def create_user(username, password, role="user"):
    conn = get_connection()
    cursor = conn.cursor()

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    cursor.execute("""
        INSERT OR IGNORE INTO users
        (username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        username,
        password_hash,
        role,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def validate_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT password_hash, role
        FROM users
        WHERE username = ?
    """, (username,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    stored_hash, role = row
    return role if bcrypt.checkpw(password.encode(), stored_hash) else None

# =====================================================
# PREDICTIONS
# =====================================================
def insert_prediction(username, condition, result, confidence, summary):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (username, condition, result, confidence, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        username,
        condition,
        result,
        confidence,
        summary,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def fetch_predictions(username=None):
    conn = get_connection()
    cursor = conn.cursor()

    if username:
        cursor.execute("""
            SELECT condition, result, confidence, created_at
            FROM predictions
            WHERE username = ?
            ORDER BY created_at DESC
        """, (username,))
    else:
        cursor.execute("""
            SELECT username, condition, result, confidence, created_at
            FROM predictions
            ORDER BY created_at DESC
        """)

    rows = cursor.fetchall()
    conn.close()
    return rows

# =====================================================
# ADMIN HELPERS
# =====================================================
def fetch_all_users():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, role, created_at
        FROM users
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, condition, result, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

# =====================================================
# CONTACT MESSAGES
# =====================================================
def insert_message(name, email, subject, message):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO messages
        (name, email, subject, message, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        email,
        subject,
        message,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def fetch_messages():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, email, subject, message, created_at
        FROM messages
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

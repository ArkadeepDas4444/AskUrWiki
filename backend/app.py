import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from rag_service import ask_question

app = Flask(__name__)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
CORS(app, resources={r"/chat": {"origins": allowed_origins}})

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.getenv("SECURITY_DB_PATH", RUNTIME_DIR / "security.db"))
DB_LOCK = threading.Lock()

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "5"))
RATE_LIMIT_PER_DAY_PER_IP = int(os.getenv("RATE_LIMIT_PER_DAY_PER_IP", "20"))
GLOBAL_DAILY_CAP = int(os.getenv("GLOBAL_DAILY_CAP", "200"))
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
BYPASS_LIMIT_IPS = {
    ip.strip()
    for ip in os.getenv("BYPASS_LIMIT_IPS", "").split(",")
    if ip.strip()
}
BYPASS_TURNSTILE_IPS = {
    ip.strip()
    for ip in os.getenv("BYPASS_TURNSTILE_IPS", "").split(",")
    if ip.strip()
}

def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS request_usage (
                ip_address TEXT NOT NULL,
                day_bucket TEXT NOT NULL,
                minute_bucket TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (ip_address, day_bucket, minute_bucket)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS global_usage (
                day_bucket TEXT PRIMARY KEY,
                request_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.commit()

def get_client_ip():
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    return request.headers.get("X-Real-IP", request.remote_addr or "unknown")

def get_time_buckets():
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%dT%H:%M")


def is_limit_bypass_ip(ip_address):
    return ip_address in BYPASS_LIMIT_IPS


def is_turnstile_bypass_ip(ip_address):
    return ip_address in BYPASS_TURNSTILE_IPS

def get_usage_snapshot(ip_address):
    day_bucket, minute_bucket = get_time_buckets()

    with DB_LOCK:
        with get_db_connection() as connection:
            minute_row = connection.execute(
                """
                SELECT request_count
                FROM request_usage
                WHERE ip_address = ? AND day_bucket = ? AND minute_bucket = ?
                """,
                (ip_address, day_bucket, minute_bucket),
            ).fetchone()

            daily_row = connection.execute(
                """
                SELECT COALESCE(SUM(request_count), 0) AS total
                FROM request_usage
                WHERE ip_address = ? AND day_bucket = ?
                """,
                (ip_address, day_bucket),
            ).fetchone()

            global_row = connection.execute(
                """
                SELECT request_count
                FROM global_usage
                WHERE day_bucket = ?
                """,
                (day_bucket,),
            ).fetchone()

    return {
        "minute_count": minute_row["request_count"] if minute_row else 0,
        "daily_count": daily_row["total"] if daily_row else 0,
        "global_daily_count": global_row["request_count"] if global_row else 0,
        "day_bucket": day_bucket,
        "minute_bucket": minute_bucket,
    }

def record_usage(ip_address, day_bucket, minute_bucket):
    with DB_LOCK:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO request_usage (ip_address, day_bucket, minute_bucket, request_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(ip_address, day_bucket, minute_bucket)
                DO UPDATE SET request_count = request_count + 1
                """,
                (ip_address, day_bucket, minute_bucket),
            )
            connection.execute(
                """
                INSERT INTO global_usage (day_bucket, request_count)
                VALUES (?, 1)
                ON CONFLICT(day_bucket)
                DO UPDATE SET request_count = request_count + 1
                """,
                (day_bucket,),
            )
            connection.commit()

def verify_turnstile_token(token, ip_address):
    if not TURNSTILE_SECRET_KEY:
        raise RuntimeError("TURNSTILE_SECRET_KEY is not configured on the server.")

    response = requests.post(
        "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        data={
            "secret": TURNSTILE_SECRET_KEY,
            "response": token,
            "remoteip": ip_address,
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("success", False)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    turnstile_token = (data.get("turnstileToken") or "").strip()
    ip_address = get_client_ip()

    if not question:
        return jsonify({"response": "Please enter a question before sending."}), 400

    if len(question) > 500:
        return jsonify({"response": "Please keep the question under 500 characters for this demo."}), 400

    if not turnstile_token:
        if not is_turnstile_bypass_ip(ip_address):
            return jsonify({"response": "Please complete the verification challenge before chatting."}), 400

    usage = get_usage_snapshot(ip_address)

    if not is_limit_bypass_ip(ip_address):
        if usage["minute_count"] >= RATE_LIMIT_PER_MINUTE:
            return jsonify({"response": "Too many requests from this IP. Please wait a minute and try again."}), 429

        if usage["daily_count"] >= RATE_LIMIT_PER_DAY_PER_IP:
            return jsonify({"response": "This IP has reached the daily demo limit. Please try again tomorrow."}), 429

        if usage["global_daily_count"] >= GLOBAL_DAILY_CAP:
            return jsonify({"response": "Daily demo limit reached. Please try again tomorrow."}), 429

    try:
        if not is_turnstile_bypass_ip(ip_address):
            if not verify_turnstile_token(turnstile_token, ip_address):
                return jsonify({"response": "Verification failed. Please refresh the challenge and try again."}), 403

        response = ask_question(question)
        if not is_limit_bypass_ip(ip_address):
            record_usage(ip_address, usage["day_bucket"], usage["minute_bucket"])

        return jsonify({"response": response}), 200

    except requests.RequestException:
        return jsonify({"response": "Security verification is temporarily unavailable. Please try again shortly."}), 503
    except Exception as error:
        return jsonify({"response": str(error)}), 500

init_db()

if __name__ == "__main__":
    app.run(debug=True)

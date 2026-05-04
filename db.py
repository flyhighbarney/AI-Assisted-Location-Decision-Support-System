import os
import pyodbc


def get_connection():
    conn_str = os.getenv("SQL_CONNECTION_STRING")

    if not conn_str:
        raise RuntimeError(
            "Missing SQL_CONNECTION_STRING. Add it in Azure App Service > Environment variables > Connection strings."
        )

    return pyodbc.connect(conn_str, timeout=30)


def test_connection():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        row = cursor.fetchone()
        return row[0] == 1

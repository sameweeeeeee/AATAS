import sqlite3
import os

db_path = "/Users/erianeetiekhong/Documents/AATAS_PROJECT/data/aatas.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memories")
    print(cursor.fetchall())
    conn.close()
else:
    print("DB not found")

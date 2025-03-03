import json
import psycopg2

# Load JSON file
with open("hash_database.json", "r") as f:
    data = json.load(f)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="songdata",
    user="postgres",
    password="livinglegend",
    host="35.182.230.172"
)
cur = conn.cursor()

# Insert data into PostgreSQL
for key, value in data.items():
    cur.execute("""
        INSERT INTO hash_data (hash_key, hash_value)
        VALUES (%s, %s)
        ON CONFLICT (hash_key) DO UPDATE SET hash_value = EXCLUDED.hash_value;
    """, (key, value))

# Commit changes & close connection
conn.commit()
cur.close()
conn.close()

print("JSON data imported into PostgreSQL successfully!")

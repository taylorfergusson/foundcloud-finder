import psycopg2
import json
from collections import defaultdict

# --- DATABASE CONNECTION SETTINGS ---
DB_NAME = "song_hashes"
DB_USER = "postgres"
DB_PASSWORD = "l1v1ngl3g3nd??"
DB_HOST = "localhost"  # Change to your server if needed
DB_PORT = "5432"

print("lOADING DATABASE")
with open('hash_database.json', 'r') as f:
    database = json.load(f)

print("TRANSFORMING DICTIONARY")
# --- 1. TRANSFORM DICTIONARY FORMAT ---
hash_to_songs = defaultdict(set)
index = 1
for song_url, hash_list in database.items():
    print("SONG NO:", index)
    for h in hash_list:
        hash_to_songs[h].add(song_url)
    index += 1

print("STORING IN POSTGRESQL")
# --- 2. STORE IN POSTGRESQL ---
try:
    print("Connecting")
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    print("Creating table")
    # Create the table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS song_hashes (
            hash TEXT PRIMARY KEY,
            song_names TEXT[] -- Store multiple song URLs
        );
    """)
    print("Committing")
    conn.commit()

    print("Inserting data")
    n = len(hash_to_songs)
    # Insert data
    insert_query = "INSERT INTO song_hashes (hash, song_names) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING"
    # data = [(h, list(songs)) for h, songs in hash_to_songs.items()]
    data = []
    i = 1
    for h, songs in hash_to_songs.items():
        print(i, '/', n)
        data.append((h, list(songs)))
        i += 1

    print("Executing")
    cur.executemany(insert_query, data)
    print("Committing")
    conn.commit()

    print("Migration completed successfully!")

except Exception as e:
    print("Error:", e)
finally:
    cur.close()
    conn.close()

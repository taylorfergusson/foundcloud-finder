import psycopg2
import json
from collections import defaultdict

# --- DATABASE CONNECTION SETTINGS ---
DB_NAME = "foundcloud_db"
DB_USER = "postgres"
DB_PASSWORD = "l1v1ngl3g3nd??"
DB_HOST = "localhost"  # Change to your server if needed
DB_PORT = "5432"


try:
    print("Connecting")
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    print("Creating table")
    # Create the table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS song_info (
            song_url TEXT PRIMARY KEY,
            artwork_url TEXT,
            title TEXT,
            username TEXT,
            duration INT,
            bpm INT
        );
    """)
    print("Committing")
    conn.commit()

    # print("Inserting data")
    # # Insert data
    # insert_query = """
    #     INSERT INTO song_hashes (hash, song_names) 
    #     VALUES (%s, %s) 
    #     ON CONFLICT (hash) 
    #     DO UPDATE 
    #     SET song_names = 
    #         CASE 
    #             WHEN NOT song_hashes.song_names @> EXCLUDED.song_names THEN 
    #                 song_hashes.song_names || EXCLUDED.song_names
    #             ELSE
    #                 song_hashes.song_names
    #         END;
    #     """
    # data = []
    # for h, songs in hash_to_songs.items():
    #     data.append((h, list(songs)))

    # print("Executing")
    # cur.executemany(insert_query, data)
    # conn.commit()

    # print("Migration completed successfully!")

except Exception as e:
    print("Error:", e)
finally:
    cur.close()
    conn.close()

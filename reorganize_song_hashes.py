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
    cursor = conn.cursor()

    print("Executing")
    # Execute the SQL query
    #cursor.execute("SELECT hash, song_names FROM song_hashes WHERE array_length(song_names, 1) > 1 LIMIT 10")
    cursor.execute("SELECT * FROM song_hashes")

    print("Fetching")
    # Fetch results
    results = cursor.fetchall()
    print("Length of results:", len(results))

    songs = set()
    # Print results
    for row in results:
        for url in row[1]:
            songs.add(url)

    songs_list = list(songs)
    songs_list.sort()
    songs_list.sort(key=len)

    with open('songs.txt', 'w') as f:
        for url in songs_list:
            f.write(url + '\n')

except Exception as e:
    print("Error:", e)
finally:
    cursor.close()
    conn.close()

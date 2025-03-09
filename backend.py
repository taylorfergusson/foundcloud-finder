import numpy as np
import os
import shutil
import librosa
import psycopg2
from math import floor
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

SAMPLE_RATE = 44100 
N_FFT = 4096
HOP_LENGTH = N_FFT // 4
LOW_CUT = 0.15
HIGH_CUT = 0.25
TRIM_START = 0
TRIM_END = 0

def get_audio_samples(filepath, sr=SAMPLE_RATE, trim_start=TRIM_START, trim_end=TRIM_END):
    try:
        samples, _ = librosa.load(filepath, mono=True, sr=sr)
        samples = librosa.util.normalize(samples)
        samples = samples[(round(sr*trim_start)):-(round(sr*trim_end)+1)]
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Possible file corruption or format issue.")
        return np.array([])
    return samples

def get_spectrogram(samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Compute the spectrogram
    Sxx = librosa.feature.melspectrogram(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return Sxx

def extract_key_points(Sxx, low_cut=LOW_CUT, high_cut=HIGH_CUT):
    bins = Sxx.shape[0]
    
    key_points = []
    for t_idx in range(len(Sxx[1])):
        # Get the frequency bins with the highest magnitude in the current time slice
        slice_magnitudes = Sxx[:, t_idx]
        slice_magnitudes[0] = 0

        f_idx = floor(bins * low_cut)
        frame = (bins - f_idx - floor(bins * high_cut)) // 4 

        top_freqs = [0, 0, 0, 0]

        for i in range(4):
            for j in range(frame):
                magnitude = slice_magnitudes[f_idx]
                if magnitude > slice_magnitudes[top_freqs[i]]:
                    top_freqs[i] = f_idx

                f_idx += 1

        if 0 not in top_freqs:
            key_points.append(top_freqs)
            
    return key_points

def generate_hashes(key_points):
    hashes = []
    for points in key_points:
        hash_value = ''.join([str(p) for p in points])
        hashes.append(int(hash_value))
    return hashes


def get_matches(query_hashes):
    try:
        conn = psycopg2.connect(
            dbname="foundcloud_db",
            user="postgres",
            password="l1v1ngl3g3nd??",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        print("TYPE:", type(query_hashes[0]))

        # Find all matching songs for given query hashes
        query = "SELECT song_urls FROM song_hashes WHERE hash = ANY(%s)"
        cur.execute(query, (query_hashes,))

        # Aggregate match counts
        match_counts = defaultdict(int)
        for row in cur.fetchall():
            for song in row[0]:  # song_urls is a list
                match_counts[song] += 1

        conn.close()

        # Return top 3 matches
        return sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    except Exception as e:
        print("Database error for song_hashes:", e)
        return []
    
def get_song_info(url):
    try:
        conn = psycopg2.connect(
            dbname="foundcloud_db",
            user="postgres",
            password="l1v1ngl3g3nd??",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        # Find all matching songs for given query hashes
        query = "SELECT * FROM song_info WHERE song_url = %s"
        cur.execute(query, (url,))

        row = cur.fetchone()

        song_info = {}
        song_info['songURL'] = row[0]
        song_info['artworkURL'] = row[1]
        song_info['title'] = row[2]
        song_info['username'] = row[3]
        song_info['duration'] = row[4]
        song_info['bpm'] = row[5]

        return song_info

    except Exception as e:
        print("Database error for song_info:", e)
        return {'songURL': url, 'artworkURL': 'https://i.imgur.com/T5D5wxK.jpeg', 'title': 'Unknown Song', 'username': 'Match still found -- click tomato', 'duration': 0, 'bpm': 0}

def check_snippet(filepath):
    # Load the MP3 file
    samples = get_audio_samples(filepath)

    # Convert samples to float32 for librosa
    # samples_float = samples.astype(np.float32) / np.max(np.abs(samples))  # Normalize audio
    # samples_float = librosa.effects.time_stretch(samples_float, rate=1.0)
    # samples_float = librosa.effects.pitch_shift(samples_float, sr=SAMPLE_RATE, n_steps=0)
    # samples = (samples_float * np.max(np.abs(samples))).astype(np.int16)  # Convert back to int16

    Sxx = get_spectrogram(samples)
    key_points = extract_key_points(Sxx)
    song_hashes = generate_hashes(key_points)
    print(song_hashes[0:3])

    matches = get_matches(song_hashes)

    for song_name, num_matches in matches:
        print(f'Song: {song_name}, Matches: {num_matches}')

    if len(matches) > 1:
        confidence = round(100 - ((matches[1][1] / matches[0][1]) * 50) - (100 / matches[0][1]))
    else:
        confidence = 100

    #return matches[0][0], max(confidence, 0)
    return 'https://soundcloud.com/user21041984001/home-made-polysynth', 0

app = FastAPI()

# Allowed frontend origins
allowed_origins = [
    "http://127.0.0.1:5500",   # Local frontend (Live Server)
    "http://localhost:5500",   # Alternative local frontend
    "https://foundcloud.taylorfergusson.com"  # Deployed frontend
]

# Add CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Directory to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get('/health/')
async def health_check():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    try:
        print("Received request")

        new_filename = request.client.host.replace(".", "-") + datetime.now().strftime("_%y-%m-%d_%H-%m-%S") + file.filename[3:]

        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER, new_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file
        result, confidence = check_snippet(file_path)  # Now we pass the file path

        info = get_song_info(result)
        info["confidence"] = "Confidence: " + str(confidence) + "%"

        return JSONResponse(content=info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    file_path = './sueno.mp3'
    url = 'https://soundcloud.com/user21041984001/home-made-polysynth'
    info = get_song_info(url)
    result, confidence = check_snippet(file_path)  # Now we pass the file path
    print("RESULT:", result)
    print("CONFIDENCE:", confidence)
    info["confidence"] = "Confidence: " + str(confidence) + "%"
    print("INFO:", info)
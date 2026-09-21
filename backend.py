import hashlib
import os
import pathlib
import shutil
from collections import defaultdict
from datetime import datetime
from warnings import filterwarnings

import librosa
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure, iterate_structure

filterwarnings("ignore")

SAMPLE_RATE = 44100
N_FFT = 4096
HOP_LENGTH = N_FFT // 4
N_MELS = 256

DEFAULT_FAN_VALUE = 5
DEFAULT_AMP_MIN = 10
CONNECTIVITY_MASK = 2
PEAK_NEIGHBORHOOD_SIZE = 10
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 16

MIN_FREQ = 45
MAX_FREQ = 205

#FALLBACK_IMG_URL = 'https://i.imgur.com/T5D5wxK.jpeg'

# DB config comes from environment variables instead of being hardcoded in
# source. Locally, copy .env.example to .env and fill it in (it's already
# gitignored). On the server, these are set in /etc/foundcloud/foundcloud.env
# and loaded by the systemd unit -- never commit a real .env file.
DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    #"sslmode": "require"
}

# match_counts used to be a single module-level dict shared by every
# request, which meant one visitor's matches could leak into another's
# results. It's now tracked per-client (see session_matches / upload_audio).
#match_counts = defaultdict(int)
session_matches = {}

def get_audio_samples(filepath, sr=SAMPLE_RATE):
    try:
        samples, _ = librosa.load(filepath, mono=True, sr=sr)
        samples = librosa.util.normalize(samples)
    except ValueError as e:
        print(f"ValueError: {e} -- Possible file corruption or format issue")
        # Return silent 1 second
        return np.zeros(SAMPLE_RATE)
    return samples

def get_tempo(samples, sr=SAMPLE_RATE):
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    int_tempo = round(float(np.atleast_1d(tempo)[0]))
    return int_tempo

def get_spectrogram(samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    Sxx = librosa.feature.melspectrogram(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return Sxx

def extract_peaks(Sxx, dam=DEFAULT_AMP_MIN, cm=CONNECTIVITY_MASK, pns=PEAK_NEIGHBORHOOD_SIZE):
    struct = generate_binary_structure(2, cm)
    neighborhood = iterate_structure(struct, pns)

    # find local maxima using our filter mask
    local_max = maximum_filter(Sxx, footprint=neighborhood) == Sxx

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (Sxx == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = Sxx[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > dam)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    return list(zip(freqs_filter, times_filter))

def generate_hashes(peaks, tempo, dfv=DEFAULT_FAN_VALUE, min_hst=MIN_HASH_TIME_DELTA, max_hst=MAX_HASH_TIME_DELTA, fr=FINGERPRINT_REDUCTION, min_freq=MIN_FREQ, max_freq=MAX_FREQ):
    # frequencies are in the first position of the tuples
    idx_freq = 0
    # times are in the second position of the tuples
    idx_time = 1

    hashes = set()
    for i in range(len(peaks)):
        for j in range(1, dfv):
            if (i + j) < len(peaks):

                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if min_hst <= t_delta <= max_hst and min_freq <= freq1 <= max_freq and min_freq <= freq2 <= max_freq:
                    h = hashlib.sha1(f"{str(tempo)}|{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))
                    hex_hash = h.hexdigest()
                    int_hash = int(hex_hash, 16) % (10**fr)
                    hashes.add(int_hash)

    return list(hashes)

def get_matches(query_hashes, match_counts):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                query = "SELECT song_paths FROM song_hashes WHERE hash = ANY(%s)"
                cur.execute(query, (list(query_hashes),))

                # Aggregate match counts
                for row in cur.fetchall():
                    for song in row[0]:
                        match_counts[song] += 1

        # Return top 3 matches
        return sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    except Exception as e:
        print(f"Database error for song_hashes: {e}")
        return []

def get_song_info(song_path):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                query = "SELECT * FROM song_info WHERE song_path = %s"
                cur.execute(query, (song_path,))

                row = cur.fetchone()

        song_info = {'song_path': row[0], 'artwork_path': row[1], 'title': row[2], 'username': row[3], 'duration': row[4], 'tempo': row[5], 'key': row[6], 'num_hashes': row[7]}

        return song_info

    except Exception as e:
        print(f"Database error for song_info: {e}")
        return None

def check_snippet(filepath, match_counts):
    # Load the MP3 file
    samples = get_audio_samples(filepath)

    # Convert samples to float32 for librosa
    # samples_float = samples.astype(np.float32) / np.max(np.abs(samples))  # Normalize audio
    # samples_float = librosa.effects.time_stretch(samples_float, rate=1.0)
    # samples_float = librosa.effects.pitch_shift(samples_float, sr=SAMPLE_RATE, n_steps=0)
    # samples = (samples_float * np.max(np.abs(samples))).astype(np.int16)  # Convert back to int16

    Sxx = get_spectrogram(samples)
    tempo = get_tempo(samples)
    peaks = extract_peaks(Sxx)
    song_hashes = generate_hashes(peaks, tempo)
    matches = get_matches(song_hashes, match_counts)

    for song_name, num_matches in matches:
        print(f'Song: {song_name}, Matches: {num_matches}')

    if len(matches) > 1:
        confidence = round(100 * (1 - (matches[1][1] / matches[0][1])))
    else:
        confidence = 0


    if len(matches) == 0:
        match = ''
    else:
        match = matches[0][0]

    return match, confidence

app = FastAPI()

# Allowed frontend origins
allowed_origins = [
    "http://127.0.0.1:5500",   # Local frontend (Live Server)
    "http://localhost:5500",   # Alternative local frontend
    os.environ.get("FRONTEND_ORIGIN", "https://foundcloud.taylorfergusson.com")  # Deployed frontend
]

# Add CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allow all HTTP methods
    allow_headers=["Authorization", "Content-Type"],  # Allow all headers
)

UPLOAD_FOLDER = pathlib.Path(os.environ.get("UPLOAD_FOLDER", "uploads")).resolve()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".webm"}
user_matches = defaultdict(list)

def allowed_file(filename):
    return pathlib.Path(filename).suffix.lower() in ALLOWED_EXTS

@app.get('/health/')
async def health_check():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_audio(request: Request, file: UploadFile = File(...), clipNum: str = Form(None)):
    # Match counts are scoped per client so concurrent visitors never see
    # each other's results (this used to be a single shared global dict).
    client_ip = request.client.host
    if clipNum == '1' or client_ip not in session_matches:
        session_matches[client_ip] = defaultdict(int)
    match_counts = session_matches[client_ip]

    try:
        print(f"Received request #{clipNum}")
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        if clipNum == '1':
            match_counts.clear()  # Instead of reassigning it

        filename = request.client.host.replace(".", "-") + datetime.now().strftime("_%y-%m-%d_%H-%M-%S") + file.filename[3:]
        # Save the file
        filepath = UPLOAD_FOLDER / filename
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the file
        result, confidence = check_snippet(str(filepath), match_counts)  # Now we pass the file path
        print(f"Result: {result}")
        print(f"Confidence: {confidence}")
        if not result or confidence < 20 and int(clipNum) < 4:
            print("No solid matches found")
            return JSONResponse(content={})
        info = get_song_info(result)
        info["confidence"] = "Confidence: " + str(confidence) + "%"
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    filepath = './sueno.webm'
    result, confidence = check_snippet(filepath, defaultdict(int))  # Now we pass the file path

    info = get_song_info(result)
    info["confidence"] = "Confidence: " + str(confidence) + "%"

    print(info)

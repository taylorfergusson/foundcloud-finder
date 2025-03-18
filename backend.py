import hashlib
import logging
import pathlib
import shutil
from collections import defaultdict
from datetime import datetime
from warnings import filterwarnings

import librosa
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure, iterate_structure

filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

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

# DB_CONFIG = {
#     "dbname": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASS"),
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT"),
#     "sslmode": "require"
# }

DB_CONFIG = {
    "dbname": 'foundcloud_db',
    "user": 'postgres',
    "password": 'l1v1ngl3g3nd??',
    "host": 'localhost',
    "port": 5432,
    #"sslmode": "require"
}


def get_audio_samples(filepath, sr=SAMPLE_RATE):
    try:
        print("Loading")
        samples, _ = librosa.load(filepath, mono=True, sr=sr)
        print("Normalizing")
        samples = librosa.util.normalize(samples)
        print("Done!")
    except ValueError as e:
        print("GOT ERROR IN GAS")
        logging.error(f"ValueError: {e} -- Possible file corruption or format issue")
        # Return silent 1 second
        return np.zeros(SAMPLE_RATE)
    print("Returning")
    return samples

def get_tempo(samples, sr=SAMPLE_RATE):
    onset_env = librosa.onset.onset_strength(y=samples, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    int_tempo = round(float(tempo))
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

def get_matches(query_hashes):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                query = "SELECT song_paths FROM song_hashes WHERE hash = ANY(%s)"
                cur.execute(query, (list(query_hashes),))

                # Aggregate match counts
                match_counts = defaultdict(int)
                for row in cur.fetchall():
                    for song in row[0]:
                        match_counts[song] += 1

        # Return top 3 matches
        return sorted(match_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    except Exception as e:
        logging.error(f"Database error for song_hashes: {e}")
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
        logging.error(f"Database error for song_info: {e}")
        return {'song_path': song_path, 'artwork_path': '', 'title': 'Unknown Song', 'username': 'Match still found -- click tomato', 'duration': 0, 'tempo': 0, 'key':'', 'num_hashes': 0}

def check_snippet(filepath):
    # Load the MP3 file
    print("Getting audio samples")
    samples = get_audio_samples(filepath)

    # Convert samples to float32 for librosa
    # samples_float = samples.astype(np.float32) / np.max(np.abs(samples))  # Normalize audio
    # samples_float = librosa.effects.time_stretch(samples_float, rate=1.0)
    # samples_float = librosa.effects.pitch_shift(samples_float, sr=SAMPLE_RATE, n_steps=0)
    # samples = (samples_float * np.max(np.abs(samples))).astype(np.int16)  # Convert back to int16

    print("Getting spect")
    Sxx = get_spectrogram(samples)
    tempo = get_tempo(samples)
    peaks = extract_peaks(Sxx)
    song_hashes = generate_hashes(peaks, tempo)
    matches = get_matches(song_hashes)

    for song_name, num_matches in matches:
        logging.info(f'Song: {song_name}, Matches: {num_matches}')

    if len(matches) > 1:
        confidence = round(100 * (1 - (matches[1][1] / matches[0][1])))
    else:
        confidence = 100


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
    "https://foundcloud.taylorfergusson.com"  # Deployed frontend
]

# Add CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Allow all HTTP methods
    allow_headers=["Authorization", "Content-Type"],  # Allow all headers
)

UPLOAD_FOLDER = pathlib.Path("uploads").resolve()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".webm"}

def allowed_file(filename):
    return pathlib.Path(filename).suffix.lower() in ALLOWED_EXTS

@app.get('/health/')
async def health_check():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    try:
        logging.info("Received request")
        print(file.filename)
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        filename = request.client.host.replace(".", "-") + datetime.now().strftime("_%y-%m-%d_%H-%M-%S") + file.filename[3:]
        # Save the file
        print(filename)
        filepath = UPLOAD_FOLDER / filename
        print("Opening file as buffer")
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Process the file
        print("Processing file")
        result, confidence = check_snippet(str(filepath))  # Now we pass the file path
        print("Getting song info")
        info = get_song_info(result)
        info["confidence"] = "Confidence: " + str(confidence) + "%"
        return JSONResponse(content=info)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    filepath = './sueno.webm'
    result, confidence = check_snippet(filepath)  # Now we pass the file path

    info = get_song_info(result)
    info["confidence"] = "Confidence: " + str(confidence) + "%"

    logging.info(info)
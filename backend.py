import numpy as np
import os
import shutil
import librosa
import psycopg2
import hashlib
from collections import defaultdict
from warnings import filterwarnings
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure, iterate_structure
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

filterwarnings("ignore")

SAMPLE_RATE = 44100
N_FFT = 4096
HOP_LENGTH = N_FFT // 4

DEFAULT_FAN_VALUE = 10
DEFAULT_AMP_MIN = 6
CONNECTIVITY_MASK = 2
PEAK_NEIGHBORHOOD_SIZE = 7
MIN_HASH_TIME_DELTA = 11
MAX_HASH_TIME_DELTA = 119
FINGERPRINT_REDUCTION = 16

def get_audio_samples(filepath, sr=SAMPLE_RATE):
    try:
        samples, _ = librosa.load(filepath, mono=True, sr=sr)
        samples = librosa.util.normalize(samples)
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Possible file corruption or format issue.")
        return np.array([])
    return samples

def get_spectrogram(samples, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Compute the spectrogram
    Sxx = librosa.feature.melspectrogram(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return Sxx

def extract_key_points(Sxx, dam=DEFAULT_AMP_MIN, cm=CONNECTIVITY_MASK, pns=PEAK_NEIGHBORHOOD_SIZE):
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

def generate_hashes(peaks, dfv=DEFAULT_FAN_VALUE, min_hst=MIN_HASH_TIME_DELTA, max_hst=MAX_HASH_TIME_DELTA, fr=FINGERPRINT_REDUCTION):
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

                    if min_hst <= t_delta <= max_hst:
                        h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))
                        hex_hash = h.hexdigest()  # Get the full hex digest
                        # Convert entire hex hash to an integer in one step
                        int_hash = int(hex_hash, 16)  
                        # Extract only the first `fr` digits
                        hash = int(str(int_hash)[:fr])
                        hashes.add(hash)

        return list(hashes)

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

    matches = get_matches(song_hashes)

    for song_name, num_matches in matches:
        print(f'Song: {song_name}, Matches: {num_matches}')

    if len(matches) > 1:
        confidence = round(100 - ((matches[1][1] / matches[0][1]) * 50) - (100 / matches[0][1]))
    else:
        confidence = 100

    if len(matches) == 0:
        match = ''
    else:
        match = matches[0][0]

    return match, max(confidence, 0)

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
    file_path = './sueno.webm'
    result, confidence = check_snippet(file_path)  # Now we pass the file path

    info = get_song_info(result)
    info["confidence"] = "Confidence: " + str(confidence) + "%"

    print(info)
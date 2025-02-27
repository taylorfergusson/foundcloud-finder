import subprocess
import numpy as np
import os
import shutil
import json
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

SAMPLE_RATE = 44100 

if os.path.exists('hash_database.json'):
    # Open and load the JSON data if the file exists
    print("Loading database...")
    with open('hash_database.json', 'r') as f:
        hash_database = json.load(f)
    print("Finished loading database.")
else:
    hash_database = {}

def get_audio_samples(filepath):
    samples, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    return samples

def get_spectrogram(samples):
    # Compute the spectrogram
    S = librosa.stft(samples, n_fft=4096, hop_length=2048)
    # Convert to power spectrogram (magnitude squared)
    Sxx = np.abs(S)**2
    return Sxx

def extract_key_points(Sxx):
    key_points = []
    for t_idx in range(len(Sxx[1])):
        # Get the frequency bins with the highest magnitude in the current time slice
        slice_magnitudes = Sxx[:, t_idx]
        slice_magnitudes[0] = 0

        top_freqs = [0, 0, 0, 0]

        index = 0
        for f_idx in range(18, 466):
            if (f_idx - 18) % 113 == 112:
                index += 1

            magnitude = slice_magnitudes[f_idx]
            if magnitude > slice_magnitudes[top_freqs[index]]:
                top_freqs[index] = f_idx

        if 0 not in top_freqs:
            key_points.append(top_freqs)
            
    return key_points

def generate_hashes(key_points):
    hashes = []
    for points in key_points:
        hash_value = ' '.join([str(p) for p in points])
        hashes.append(hash_value)
    return hashes

def get_matches(query_hashes, database):
    matches = []
    for song_name, song_hashes in database.items():
        common_hashes = len(set(query_hashes).intersection(song_hashes))
        matches.append((song_name, common_hashes))

    matches.sort(key=lambda x: x[1], reverse=True)

    top_matches = matches[:3]
    
    return top_matches

def check_snippet(filepath):
    # Load the MP3 file
    samples = get_audio_samples(filepath)

    Sxx = get_spectrogram(samples)
    key_points = extract_key_points(Sxx)
    song_hashes = generate_hashes(key_points)

    matches = get_matches(song_hashes, hash_database)
    return matches[0][0]
    # top_matches = []
    # for song_name, num_matches in matches:
    #     top_matches.append(song_name)
    #     print(f'Song: {song_name}, Matches: {num_matches}')

    # if matches[0][1] > 5 and matches[0][1] > matches[1][1]+4:
    #     print("PASSED")
    #     return matches[0][0]
    # else:
    #     return ''

def download_song_info(url):
    print("URL IS....", url)
    result = subprocess.run(['node', 'download_info.js', url], capture_output=True, text=True)
    print("Ok got the result now.....")
    if result.returncode == 0:
        print(result.stdout)
        return json.loads(result.stdout)
    else:
        print(f"Error: {url} -- General download error")
        print(result.stdout)
        print(result.stderr)
        return {'songURL': '', 'artworkURL': '', 'title': '', 'username': ''}
    

app = FastAPI()
# Directory to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    print("UPLOADING AUDIO!!!!")
    try:
        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        result = check_snippet(file_path)  # Now we pass the file path
        print(result)
        info = download_song_info(result)
        return JSONResponse(content=info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
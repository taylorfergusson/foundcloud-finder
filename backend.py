import subprocess
import numpy as np
from scipy.signal import spectrogram
import os
import json
import librosa
from flask import Flask, request, jsonify

SAMPLE_RATE = 44100

if os.path.exists('hash_database.json'):
    # Open and load the JSON data if the file exists
    print("Loading database...")
    with open('hash_database.json', 'r') as f:
        hash_database = json.load(f)
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

        # top_freqs = [18, 131, 244, 357]
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

# 4. Hashing function for matching (simple example)
def generate_hashes(key_points):
    hashes = []
    for points in key_points:
        hash_value = ' '.join([str(p) for p in points])
        hashes.append(hash_value)
    return hashes

# 6. Matching function
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

    # Convert samples to float32 for librosa
    # samples_float = samples.astype(np.float32) / np.max(np.abs(samples))  # Normalize audio
    # samples_float = librosa.effects.time_stretch(samples_float, rate=1.0)
    # samples_float = librosa.effects.pitch_shift(samples_float, sr=SAMPLE_RATE, n_steps=0)
    # samples = (samples_float * np.max(np.abs(samples))).astype(np.int16)  # Convert back to int16

    Sxx = get_spectrogram(samples)
    key_points = extract_key_points(Sxx)
    song_hashes = generate_hashes(key_points)

    matches = get_matches(song_hashes, hash_database)
    top_matches = []
    for song_name, num_matches in matches:
        top_matches.append(song_name)
        print(f'Song: {song_name}, Matches: {num_matches}')

    if matches[0][1] > 5 and matches[0][1] > matches[1][1]+4:
        print("PASSED")
        return matches[0][0]
    else:
        return ''

def download_song_info(url):
    print("URL IS....", url)
    result = subprocess.run(['node', 'download_info.js', url], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Success loading song info: {url}")
        results = json.loads(result.stdout)
        return jsonify(results)
    else:
        print(f"Error: {url} -- General download error")
        return jsonify({'artworkURL': '', 'title': '', 'username': ''})

app = Flask(__name__)

# Directory to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = check_snippet(filepath)
    print()

    # Delete the file after processing
    os.remove(filepath)

    # Return the result as a JSON response
    return jsonify({'result': result})

@app.route('/get_song_info', methods=['POST'])
def get_song_info():
    data = request.get_json()
    song_url = data.get('songURL')  # Retrieve song URL from JSON payload

    # Return the result as a JSON response
    return download_song_info(song_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

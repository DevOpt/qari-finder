import os
import uuid
import pyaudio
import wave
import warnings
import logging
import config as c
import numpy as np

from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine 
from feature_extraction import get_embedding, get_embeddings_from_list_file

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def record_audio(duration, rid):
    # Audio settings
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Stereo
    rate = 44100  # Record at 44100hz samples per second

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a new stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording...")

    frames = []  # Initialize array to store frames

    # Store data in chunks for the specified duration
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")

    # Save the recorded data as a WAV file
    output_filename = os.path.join(c.RECORDINGS_FILE, rid + ".wav")
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved as {output_filename}")


def recognize_reciter(rid):
    audio_file = os.path.join(c.RECORDINGS_FILE, rid + ".wav")
    if not os.path.exists(audio_file):
        print(f"Error audio file {audio_file} does not exist")
        exit()

    if os.path.exists(c.EMBED_LIST_FILE):
        embeds = os.listdir(c.EMBED_LIST_FILE)
    if len(embeds) is 0:
        print("No enrolled reciters found")
        exit()
    print("Loading model weights from [{}]....".format(c.MODEL_FILE))
    try:
        model = load_model(c.MODEL_FILE)
    except:
        print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
        exit()
        
    distances = {}
    print("Processing test sample....")
    print("Comparing test sample against enroll samples....")
    test_result = get_embedding(model, audio_file, c.MAX_SEC)
    test_embs = np.array(test_result.tolist())
    for emb in embeds:
        enroll_embs = np.load(os.path.join(c.EMBED_LIST_FILE,emb))
        speaker = emb.replace(".npy","")
        distance = euclidean(test_embs, enroll_embs)
        distances.update({speaker:distance})
    if min(list(distances.values()))<c.THRESHOLD:
        print("Recognized: ",min(distances, key=distances.get))
        s = dict(sorted(distances.items(), key=lambda item: item[1]))
        print("Ranked:")
        for i in s:
            print(i)
    else:
        print("Could not identify the reciter, try enrolling again with a clear voice sample")
        print("Score: ",min(list(distances.values())))
        exit()



if __name__ == "__main__":
    # Record audio for 5 seconds and save to output.wav
    duration = 10  # Duration in seconds
    rid = str(uuid.uuid4())  # Generate recording id
    record_audio(duration, rid)
    recognize_reciter(rid)

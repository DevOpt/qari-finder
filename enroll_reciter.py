#IMPORT SYSTEM FILES
import argparse
import scipy.io.wavfile as wavfile
import traceback as tb
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine 
import warnings
from keras.models import load_model
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
#IMPORT USER-DEFINED FUNCTIONS
from feature_extraction import get_embedding, get_embeddings_from_list_file
from preprocess import get_fft_spectrum
import config as c

# args() returns the args passed to the script
def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--metadata',
                       help='Reciter metadata in JSON format',
                       required=False)
    parser.add_argument( '-n', '--name',
                        help='Specify the name of the person you want to enroll',
                        required=False)
    parser.add_argument('-f', '--file',
                        help='Specify the audio file you want to enroll',
                        type=lambda fn:file_choices(("wav","flac", "json"),fn),
                       required=True)
    parser.add_argument('-b', '--bulk', 
                        action=argparse.BooleanOptionalAction)

    ret = parser.parse_args()
    return ret


def enroll(name,file):
    """Enroll a reciter with an audio file
        inputs: str (Metadata of the reciter to be enrolled)
                str (Name of the person to be enrolled and registered)
                str (Path to the audio file of the reciter to enroll)
        outputs: None"""

    print("Loading model weights from [{}]....".format(c.MODEL_FILE))
    try:
        model = load_model(c.MODEL_FILE)
    except:
        print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
        exit()
    
    try:
        print("Processing enroll sample....")
        enroll_result = get_embedding(model, file, c.MAX_SEC)
        enroll_embs = np.array(enroll_result.tolist())
        speaker = name
    except:
        print("Error processing the input audio file. Make sure the path.")
    try:
        np.save(os.path.join(c.EMBED_LIST_FILE,speaker +".npy"), enroll_embs)
        print("Succesfully enrolled the reciter")
    except:
        print("Unable to save the reciter into the database.")


def bulk_enroll(file):
    f = open(file)
    metadata = json.load(f)
    reciters_metadata = metadata['reciters']
    print("Loading model weights from [{}]....".format(c.MODEL_FILE))
    try:
        model = load_model(c.MODEL_FILE)
    except:
        print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
        exit()

    for reciter in reciters_metadata:
        name = reciter['name']
        audioFile = reciter['audioFile']
        try:
            print("Processing enroll sample....")
            enroll_result = get_embedding(model, audioFile, c.MAX_SEC)
            enroll_embs = np.array(enroll_result.tolist())
        except:
            print("Error processing the input audio file. Make sure the path.")
        try:
            np.save(os.path.join(c.EMBED_LIST_FILE,name +".npy"), enroll_embs)
            print(f'Succesfully enrolled the reciter {name}')
        except:
            print("Unable to save the reciter into the database.")
    f.close()


# Verify file extenstion
def file_choices(choices,filename):
    ext = os.path.splitext(filename)[1][1:]
    if ext not in choices:
        logging.error("file doesn't end with one of {}".format(choices))
    return filename


if __name__ == '__main__':
    try:
        args = args()
    except Exception as e:
        print('An Exception occured, make sure the file format is .wav or .flac')
        exit()
    file = args.file
    if args.bulk:
        bulk_enroll(file)
    else:
        name = args.name
        enroll(name, file)

import pyaudio
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import sounddevice as sd
from scipy.io.wavfile import write
import pathlib
import numpy as np
import pandas as pd
import os
import miniaudio
import pydub


datadir = pathlib.Path("cv-corpus")
target_dir = pathlib.Path("entestdata")


def tsv2df(datadir, lang="en", split="test"):
    """create dataframe
    """
    split = split + ".tsv"
    df = pd.read_table(datadir/lang/split, low_memory=False)
    df = df[["client_id", "path", "sentence", "locale"]]
    df["path"] = df.path.apply(lambda p: os.path.join(datadir/lang/"clips/", p))
    df["split"] = split.split(".")[0]
    print(split)
    if split=="train.tsv":
        df = df.sample(min(30000, len(df)))
    print(df.shape)
    return df


df= tsv2df(datadir)
df = df.sample(500, random_state=42)


def wav2vec2encode(file):
    audio = pydub.AudioSegment.from_mp3(file).set_frame_rate(16000)
    print(target_dir/(os.path.splitext(os.path.basename(file))[0]+".wav"))
    audio.export(target_dir/(os.path.splitext(os.path.basename(file))[0]+".wav"), format="wav")

df["dur"] = np.array([miniaudio.mp3_get_file_info(path).duration for path in df.path], np.float32)
df.groupby("split").dur.sum()/3600
df.path.apply(wav2vec2encode)

test = df.sample(1)
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


def stt(path, model=model, tokenizer=tokenizer):
    speech, rate = librosa.load(path,sr=16000)
    input_values = tokenizer(speech, return_tensors = 'pt').input_values
    #Store logits (non-normalized predictions)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    #decode the audio to generate text
    transcriptions = tokenizer.decode(predicted_ids[0])
    return transcriptions

import time
s = time.time()
df["stt_nopretrain"] = df.path.apply(stt)
print(time.time()-s)


df.to_csv("test.csv")
# def record_wav(output_file, duration=10, sr=16000):
#     "record audio and save as .wav"
#     audio = sd.rec(duration*sr, samplerate=sr, channels=1)
#     print(audio)
#     sd.wait()
#     write(output_file, sr, audio)
#     return 
# record_wav("test.wav", duration=5, sr=44100)
    
#load any audio file of your choice
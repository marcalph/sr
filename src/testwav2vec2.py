# stdlib
import os
import pathlib

# third party
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from scipy.io.wavfile import write
import pydub
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4)
pd.set_option("max_colwidth", 50)
pd.set_option("display.max_columns", 50)

# relative
from src.datautils import commonvoice_df


cv_dir = pathlib.Path("/home/marcalph/Downloads/cv-corpus-6.1-2020-12-11/")
testpath = pathlib.Path("frtestdata")


def get_test_path(path, testpath=testpath):
    """ change path from orig path to test path
    """
    return testpath/os.path.basename(path).replace("mp3", "wav")




df = commonvoice_df(cv_dir)
df["testpath"] = df.path.parallel_apply(get_test_path)

df.head()

update_path_to_test(test.path.values[0])


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


def stt(path, model=model, tokenizer=tokenizer):
    speech, rate = librosa.load(path,sr=16000)
    input_values = tokenizer(speech, return_tensors = 'pt').input_values
    # get logits 
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    # decode the audio to generate text
    transcriptions = tokenizer.decode(predicted_ids[0])
    return transcriptions

import time
s = time.time()
df["stt_nopretrain"] = df.path.parallel_apply(stt)
print(time.time()-s)


df.to_csv("test5000.csv")
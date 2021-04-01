# stdlib
import os
import pathlib
from itertools import combinations

# third party
import matplotlib.pyplot as plt
import miniaudio
import numpy as np
import pandas as pd
import seaborn as sns
import pydub
import pydub.playback

languages = """
    fr
""".split()

splits = """
    train
    test
    dev
""".split()


# intitial commonvoice dir located elsewhere
cv_dir = pathlib.Path("/home/marcalph/Downloads/cv-corpus-6.1-2020-12-11/")
test_dir = pathlib.Path("frtestdata")


def commonvoice_df(datadir, split="test", lang="fr"):
    """ load commonvoice dir
    """
    split = split + ".tsv"
    df = pd.read_table(datadir/lang/split, low_memory=False)
    df = df[["client_id", "path", "sentence", "locale"]]
    df["path"] = df.path.apply(lambda p: os.path.join(datadir/lang/"clips/", p))
    df["split"] = split.split(".")[0]
    print(split)
    if split=="train.tsv":
        df = df.sample(min(70000, len(df)))
    print(df.shape)
    return df


def wav2vec2encode(filename, target_dir=test_dir):
    """ convert .mp3 to .wav file w/ 16k frame rate
    """
    audio = pydub.AudioSegment.from_mp3(filename).set_frame_rate(16000)
    print(target_dir/(os.path.splitext(os.path.basename(filename))[0]+".wav"))
    audio.export(target_dir/(os.path.splitext(os.path.basename(filename))[0]+".wav"), format="wav")
    return


def playback(filename):
    audio = pydub.AudioSegment.from_wav(filename)
    pydub.playback.play(audio)
    return



if __name__ == "__main__":
    df = commonvoice_df(common_voice)
    df["dur"] = np.array([miniaudio.mp3_get_file_info(path).duration for path in df.path], np.float32)
    df.groupby("split").dur.sum()/3600
    df.path.apply(wav2vec2encode)


import os
from tts_infer.tts import TextToMel, MelToWav
from tts_infer.transliterate import XlitEngine
from tts_infer.num_to_word_on_sent import normalize_nums

import re
import numpy as np
from scipy.io.wavfile import write

from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize

INDIC = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

def split_sentences(paragraph, language):
    if language == "en":
        with MosesSentenceSplitter(language) as splitter:
            return splitter([paragraph])
    elif language in INDIC:
        return sentence_tokenize.sentence_split(paragraph, lang=language)
#/content/vakyansh-tts/

device='cpu'
text_to_mel = TextToMel(glow_model_dir='tts_infer/odia/glow', device=device)
mel_to_wav = MelToWav(hifi_model_dir='tts_infer/odia/hifi', device=device)

def run_tts(text, lang):
    final_text = text
    mel = text_to_mel.generate_mel(final_text)
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='temp.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)

def run_tts_paragraph(text, lang):
    audio_list = []
    split_sentences_list = split_sentences(text, language='hi')

    for sent in split_sentences_list:
        sr, audio = run_tts(sent, lang)
        audio_list.append(audio)

    concatenated_audio = np.concatenate([i for i in audio_list])
    write(filename='temp_long.wav', rate=sr, data=concatenated_audio)
    return (sr, concatenated_audio)

#_, audio = run_tts("ଆମେ ଦୁଖିତ, ଆପଣଙ୍କର ଚିନ୍ତାଧାରାକୁ ସମାଧାନ କରିବାରେ ଅସମର୍ଥ, ଆମେ ଆପଣଙ୍କ ସହ ଯୋଗାଯୋଗ କରିବାକୁ ୱାର୍କସପ୍ଦ ଦଳକୁ କହିବୁ, ତୁମର ଦିନ ଶୁଭମୟ ହଉ.", "or")
#ମୋର ନାମ ଇରଫାନ୍ |
_, audio = run_tts("ମୋର ନାମ ଇରଫାନ୍ |", "or")

import IPython.display as ipd
ipd.Audio('temp.wav')

# pip install inflect
# pip install librosa
from funasr import AutoModel
import soundfile
import os

import sklearn
import matplotlib.pyplot as plt
import librosa.display

#
# use vad, punc, spk or not as you need
# model = AutoModel(model="paraformer-en", model_revision="v2.0.4",
#                   vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#                   punc_model="ct-punc-c", punc_model_revision="v2.0.4",
#                   # spk_model="cam++", spk_model_revision="v2.0.2",
#                   )
#
# res_text = model.generate(input=r"en_test/2.mp3",
#                      batch_size_s=300)
# print(res_text)

#vad
timestamps_model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
res = timestamps_model.generate(input=r"en_test/2.mp3")
print(res)

# model = AutoModel(model="iic/speech_timestamp_prediction-v1-16k-offline", model_revision="v2.0.4")
#
# wav_file = f"en_test/2.mp3"
# text_file = f"en_test/2.txt"
# res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
# print(res)

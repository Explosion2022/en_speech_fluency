import os

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment

HERE=os.path.dirname(__file__)

audio_path = os.path.join(HERE, 'en_test', '2.mp3')
print(audio_path)
y, sr = librosa.load(audio_path)
# print(y.shape, sr)

audio_file, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
# print('Audio File:', audio_file, '\n')
# print('Audio File shape:', np.shape(audio_file))

rmse = librosa.feature.rms(y=audio_file, frame_length=256, hop_length=64)[0]

print(librosa.power_to_db(rmse**2, ref=np.max))
print(librosa.get_duration(y=y,sr=sr), librosa.get_duration(y=audio_file,sr=sr))
# plt.figure(figsize=(20, 5))
# librosa.display.waveshow(audio_file)
# plt.show()
plt.figure(figsize=(20, 5))
librosa.display.waveshow(y, sr=sr)
plt.show()
def cut_mp3(filepath):

    music = AudioSegment.from_file('exam/2023213115-04-J3.mp3')
    sound_time = music.duration_seconds
    print(f"music duration time: {sound_time}")

    # 使用切片截取, 单位毫秒， 1s -> 1000ms
    out_music = music[110000: 119000]

    # 导出
    out_music.export(out_f="2023213115-04-J3-cut-short.mp3", format='mp3')   # 可以指定bitrate为64k比特率 None为源文件

    print('done')
    pass


if __name__ == '__main__':
    src_path = audio_path  # seconds: 30 -> 3'49(229)
    #cut_mp3(filepath=src_path)


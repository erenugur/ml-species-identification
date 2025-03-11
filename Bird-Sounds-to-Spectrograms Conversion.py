import matplotlib.pyplot as plt
import numpy as np
import librosa

counter = 0

for x in range(1002):
  audio_file = f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Audio/Muscicapa striata (100 audio files)/test{x}.wav'

  y, sr = librosa.load(audio_file, duration=10)   # the duration sets the max time of the spectrogram
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
  plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Removes the white edge
  img = librosa.display.specshow(D,sr=sr)
  plt.savefig(f"/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Muscicapa striata (100 spectrograms)/test{x}.png", bbox_inches=None, pad_inches=0)

  counter += 1
  if counter % 10 == 0:
    print(f"{counter} Spectrograms Downloaded")

  # png is higher quality than jpg when saving

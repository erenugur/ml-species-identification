import matplotlib.pyplot as plt
import numpy as np
import librosa

for x in range(12):
  audio_file = f'/content/drive/My Drive/AI Club Species Identification Project/Orca_audio/orca_audio{x}.wav'

  y, sr = librosa.load(audio_file, duration=10)   # the duration sets the max time of the spectrogram
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
  plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Removes the white edge
  img = librosa.display.specshow(D,sr=sr)
  plt.savefig(f"/content/drive/My Drive/AI Club Species Identification Project/Orca_spectrograms/{x}.png", bbox_inches=None, pad_inches=0)

  # png is higher quality than jpg when saving

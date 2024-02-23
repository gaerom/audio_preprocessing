# mel-spectrogram segment 차원 확인하고 visualization
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from glob import glob
import os

mel_dir = '/home/broiron/Desktop/TPoS/dataset/mels'
mel_files = glob(os.path.join(mel_dir, '*.npy'))
mel_db = np.load(mel_files[3])
mel_db_squeezed = mel_db.squeeze() 
print(mel_db_squeezed.shape) 


plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db_squeezed, sr=44100, fmax=8000, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.tight_layout()
plt.show()

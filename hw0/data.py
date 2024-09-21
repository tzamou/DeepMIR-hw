import os
import numpy as np
import glob
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import torchaudio.transforms as T

def generate_mel_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def plot_mel_spec(datafolder:str = 'E:/Dataset/nsynth-subtrain/audio/'):
    '''

    :param datafolder: the wav data folder, e.g. E:/Dataset/nsynth-subtrain/audio/
    :return:
    '''
    def plot_mel_spectrogram(mel_spec, Instrument_name, Pitch):
        img = librosa.display.specshow(mel_spec, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
        plt.title(f"Instrument: {Instrument_name}, Pitch: {Pitch}", fontsize=17)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Hz', fontsize=15)
        plt.colorbar(img, format="%+2d dB")
        plt.savefig(f'./result/spectrogram/{Instrument_name}-{Pitch}.png')
        plt.clf()

    guitar023 = librosa.load(f'{datafolder}guitar_electronic_038-023-050.wav', sr=None)[0]
    mel_guitar023 = generate_mel_spectrogram(guitar023, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_guitar023, Instrument_name='Guitar (Electronic)', Pitch=23)

    guitar078 = librosa.load(f'{datafolder}guitar_electronic_037-078-075.wav', sr=None)[0]
    mel_guitar078 = generate_mel_spectrogram(guitar078, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_guitar078, Instrument_name='Guitar (Electronic)', Pitch=78)

    guitar040 = librosa.load(f'{datafolder}guitar_electronic_037-040-100.wav', sr=None)[0]
    mel_guitar040 = generate_mel_spectrogram(guitar040, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_guitar040, Instrument_name='Guitar (Electronic)', Pitch=40)

    mallet0106 = librosa.load(f'{datafolder}mallet_acoustic_025-106-050.wav', sr=None)[0]
    mel_mallet0106 = generate_mel_spectrogram(mallet0106, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_mallet0106, Instrument_name='Mallet (Acoustic)', Pitch=106)

    mallet030 = librosa.load(f'{datafolder}mallet_acoustic_026-030-075.wav', sr=None)[0]
    mel_mallet030 = generate_mel_spectrogram(mallet030, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_mallet030, Instrument_name='Mallet (Acoustic)', Pitch=30)

    mallet045 = librosa.load(f'{datafolder}mallet_acoustic_026-045-025.wav', sr=None)[0]
    mel_mallet045 = generate_mel_spectrogram(mallet045, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_mallet045, Instrument_name='Mallet (Acoustic)', Pitch=45)

    bass022 = librosa.load(f'{datafolder}bass_electronic_000-022-127.wav', sr=None)[0]
    mel_bass022 = generate_mel_spectrogram(bass022, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_bass022, Instrument_name='Bass (Electronic)', Pitch=22)

    bass034 = librosa.load(f'{datafolder}bass_electronic_000-034-025.wav', sr=None)[0]
    mel_bass034 = generate_mel_spectrogram(bass034, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_bass034, Instrument_name='Bass (Electronic)', Pitch=34)

    bass063 = librosa.load(f'{datafolder}bass_electronic_002-063-100.wav', sr=None)[0]
    mel_bass063 = generate_mel_spectrogram(bass063, sr=22050, n_fft=2048, hop_length=512)
    plot_mel_spectrogram(mel_bass063, Instrument_name='Bass (Electronic)', Pitch=63)


class NsynthDataset(Dataset):
    def __init__(self, datafolder, n_fft, win_length, hop_length, use_log=True):
        '''
        wav資料集
        :param datafolder: original wav path (nsynth-subtrain or nsynth-test)
        '''
        self.datafolder = datafolder
        self.files = glob.glob(f"{datafolder}/audio/*.wav")
        self.instrument_family_lst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
        self.use_log = use_log
        self.spectrogram = T.MelSpectrogram(
                           n_fft=n_fft,
                           win_length=win_length,
                           hop_length=hop_length,
                           center=True,
                           pad_mode="reflect",
                           f_min=0.0,
                           f_max=8000.0,
                           n_mels=128,
                           power=2.0)
        self.to_db = T.AmplitudeToDB()

    def get_categorical_weight(self):
        training_labels_count = np.array([5005, 3411, 4118, 4633, 4535, 4674, 4499, 4232, 3830, 4580, 4520])
        weights = 1.0 / training_labels_count
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data, sr = librosa.load(file, sr=44100)
        waveform = torch.tensor(data).unsqueeze(0)  # 加上batch dimension
        spec = self.spectrogram(waveform)
        if self.use_log:
            spec = self.to_db(spec)
        # data = librosa.load(file, sr=16000)[0]
        # mel_data = generate_mel_spectrogram(data, sr=22050)
        for index, instrument_family in enumerate(self.instrument_family_lst):
            if instrument_family in file:
                label = index

        return spec, torch.tensor(label).long()

class NsynthNpyDataset(Dataset):
    def __init__(self, datafolder, transform=None):
        '''
        Npy資料集。
        :param datafolder: 原始wav資料的路徑，使用資料集下載下來後的資料夾(nsynth-subtrain or nsynth-test)
        :param transform: pytorch對資料作前處理轉換的部分
        '''
        self.datafolder = datafolder
        self.data = np.load(f'{self.datafolder}/npy/data.npy')
        self.label = np.load(f'{self.datafolder}/npy/label.npy')
        self.instrument_family_lst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
        self.transform = transform

    def get_label_count(self):
        count = np.zeros(shape=(11, ))
        for i in self.label:
            count[i] += 1
        print(count)
        # every class in sub-trai dataset number: [5005. 3411. 4118. 4633. 4535. 4674. 4499. 4232. 3830. 4580. 4520.]
        # every class in test dataset number: [843. 269. 180. 652. 766. 202. 502. 235. 306.   0. 141.]
        return count

    def get_categorical_weight(self):
        training_labels_count = np.array([5005, 3411, 4118, 4633, 4535, 4674, 4499, 4232, 3830, 4580, 4520])
        weights = 1.0 / training_labels_count
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        return torch.from_numpy(data).unsqueeze(0), torch.tensor(label).long()

def wav2npy(datafolder: str, sr=22050, n_fft=2048, hop_length=512):
    '''
    將wav檔案讀取並取得頻譜，接著轉換為npy檔案
    :param datafolder: 原始wav資料的路徑，使用資料集下載下來後的資料夾(nsynth-subtrain or nsynth-test)
    :param data_type: 資料的名稱，subtraining或者testing
    :return:
    '''
    files = glob.glob(f"{datafolder}/audio/*.wav")
    instrument_family_lst = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    data = librosa.load(files[0], sr=sr)[0]
    mel_data_npy = generate_mel_spectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length)[np.newaxis, :, :]
    # print(mel_data_npy.shape)
    for index, instrument_family in enumerate(instrument_family_lst):
        if instrument_family in files[0]:
            label_npy = np.array([index])

    for file in tqdm(files[1: ]):
        data = librosa.load(file, sr=sr)[0]
        mel_data_npy = np.append(mel_data_npy, generate_mel_spectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length)[np.newaxis, :, :], axis=0)
        # print(mel_data_npy.shape)
        for index, instrument_family in enumerate(instrument_family_lst):
            if instrument_family in file:
                label_npy = np.append(label_npy, np.array([index]))

    # 'E:/Dataset/nsynth-subtrain/npy/subtraining_data.npy'
    os.makedirs(f"{datafolder}/npy/")
    np.save(file=f"{datafolder}/npy/data.npy", arr=mel_data_npy)
    np.save(file=f"{datafolder}/npy/label.npy", arr=label_npy)



if __name__ == '__main__':
    # plot_mel_spec(datafolder='E:/Dataset/nsynth-subtrain/audio/')

    wav2npy(datafolder='E:/Dataset/nsynth-subtrain', sr=44100,  n_fft=1024, hop_length=1024)
    wav2npy(datafolder='E:/Dataset/nsynth-test', sr=44100,  n_fft=1024, hop_length=1024)









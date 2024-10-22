import os, shutil
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

def mix_file(split='train'):
    root_path = os.path.dirname(os.getcwd())
    training_path = Path(os.path.join(root_path, f"musdb18hq/{split}"))
    folders = [f for f in training_path.iterdir() if f.is_dir()]
    for folder in tqdm(folders):

        audio_files = [f"{training_path}/{folder.name}/bass.wav",
                       f"{training_path}/{folder.name}/drums.wav",
                       f"{training_path}/{folder.name}/other.wav"]

        data, sr = librosa.load(audio_files[0], sr=44100)
        combined = data
        for file in audio_files[1:]:
            track, _ = librosa.load(file, sr=sr)
            if len(track) < len(combined):
                track = np.pad(track, (0, len(combined) - len(track)), mode='constant')
            elif len(track) > len(combined):
                combined = np.pad(combined, (0, len(track) - len(combined)), mode='constant')
            combined += track
        combined = combined / np.max(np.abs(combined))
        wavfile.write(f"{training_path}/{folder.name}/nonvocal_mixture.wav", sr, (combined * 32767).astype(np.int16))

def move_to_validation():
    val_dir = ['Actions - One Minute Smile', 'Clara Berry And Wooldog - Waltz For My Victims', 'Johnny Lokke - Promises & Lies', 'Patrick Talbot - A Reason To Leave',
               'Triviul - Angelsaint', 'Alexander Ross - Goodbye Bolero', 'Fergessen - Nos Palpitants', 'Leaf - Summerghost', 'Skelpolu - Human Mistakes', 'Young Griffo - Pennies',
               'ANiMAL - Rockshow', 'James May - On The Line', 'Meaxic - Take A Step', 'Traffic Experiment - Sirens']
    root_path = os.path.dirname(os.getcwd())
    training_path = Path(os.path.join(root_path, f"musdb18hq/train"))
    validation_path = training_path.with_name("val")

    folders = [f for f in training_path.iterdir() if f.is_dir()]
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    for folder in tqdm(folders):
        if folder.name in val_dir:
            for filename in os.listdir(f'{training_path}/{folder.name}'):
                if not os.path.exists(f'{validation_path}/{folder.name}'):
                    os.makedirs(f'{validation_path}/{folder.name}')
                src_file = os.path.join(f'{training_path}/{folder.name}', filename)
                dst_file = os.path.join(f'{validation_path}/{folder.name}', filename)
                if os.path.isdir(src_file):
                    shutil.copytree(src_file, dst_file)  # 複製資料夾
                else:
                    shutil.copy2(src_file, dst_file)
            shutil.rmtree(f'{training_path}/{folder.name}')

def get_full_npy_per_5sec(split='train'):
    root_path = os.path.dirname(os.getcwd())
    training_path = Path(os.path.join(root_path, f"musdb18hq/{split}"))

    folders = [f for f in training_path.iterdir() if f.is_dir()]
    sr = 44100*5
    all_vocals_data = np.array([])
    all_mixture_data = np.array([])

    for folder in tqdm(folders):
        audio_files = [f"{training_path}/{folder.name}/vocals.npy",
                       f"{training_path}/{folder.name}/mixture.npy"]

        vocals_data = np.load(audio_files[0])
        mixture_data = np.load(audio_files[1])
        sec = vocals_data.shape[1] // sr

        vocals_data = vocals_data[:, :sec*sr].reshape(-1, sr)
        mixture_data = mixture_data[:, :sec*sr].reshape(-1, sr)
        for i in range(vocals_data.shape[0]):
            if np.max(vocals_data[i])<0.01 and np.min(vocals_data[i])>-0.01:
                continue
            else:
                if len(all_mixture_data.shape) == 1:
                    all_vocals_data = vocals_data[i].reshape(-1, sr)
                    all_mixture_data = mixture_data[i].reshape(-1, sr)
                else:
                    all_vocals_data = np.append(all_vocals_data, vocals_data[i].reshape(-1, sr), axis=0)
                    all_mixture_data = np.append(all_mixture_data, mixture_data[i].reshape(-1, sr), axis=0)
                    # print(all_vocals_data.shape)
    np.save(file=f"{training_path}/vocals.npy", arr=all_vocals_data)
    print(all_vocals_data.shape)
    np.save(file=f"{training_path}/mixture.npy", arr=all_mixture_data)

def get_npy(split='train'):
    root_path = os.path.dirname(os.getcwd())
    training_path = Path(os.path.join(root_path, f"musdb18hq/{split}"))
    folders = [f for f in training_path.iterdir() if f.is_dir()]

    for folder in tqdm(folders):
        audio_files = [f"{training_path}/{folder.name}/vocals.wav",
                       f"{training_path}/{folder.name}/mixture.wav",
                       f"{training_path}/{folder.name}/other.wav"]

        vocals_data, sr = torchaudio.load(audio_files[0])
        if sr != 44100:
            resampler = T.Resample(orig_freq=sr, new_freq=44100)
            vocals_data = resampler(vocals_data)

        mixture_data, sr = torchaudio.load(audio_files[1])
        if sr != 44100:
            resampler = T.Resample(orig_freq=sr, new_freq=44100)
            mixture_data = resampler(mixture_data)

        np.save(file=f"{training_path}/{folder.name}/vocals.npy", arr=vocals_data[0].unsqueeze(0).numpy())
        np.save(file=f"{training_path}/{folder.name}/mixture.npy", arr=mixture_data[0].unsqueeze(0).numpy())
        try:
            other_data, sr = torchaudio.load(audio_files[2])
            if sr != 44100:
                resampler = T.Resample(orig_freq=sr, new_freq=44100)
                other_data = resampler(other_data)
            np.save(file=f"{training_path}/{folder.name}/non-vocal.npy", arr=other_data[0].unsqueeze(0).numpy())
        except:
            pass

def get_npy_inference():
    root_path = os.path.dirname(os.getcwd())
    training_path = Path(os.path.join(root_path, f"musdb18hq/inference"))
    folders = [f for f in training_path.iterdir() if f.is_dir()]

    for folder in tqdm(folders):
        audio_files = f"{training_path}/{folder.name}/mixture.wav"

        mixture_data, sr = torchaudio.load(audio_files)
        if sr != 44100:
            resampler = T.Resample(orig_freq=sr, new_freq=44100)
            mixture_data = resampler(mixture_data)

        np.save(file=f"{training_path}/{folder.name}/mixture.npy", arr=mixture_data[0].unsqueeze(0).numpy())

class Musdb19HQ(Dataset):
    def __init__(self, split='train'):
        assert split in ['train', 'test', 'val']
        self.training_path = Path(os.path.join(os.getcwd(), f"musdb18hq/{split}"))
        self.folders = [f for f in self.training_path.iterdir() if f.is_dir()]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        audio_files = [f"{self.training_path}/{self.folders[idx].name}/vocals.npy",
                       f"{self.training_path}/{self.folders[idx].name}/mixture.npy",
                       f"{self.training_path}/{self.folders[idx].name}/non-vocal.npy"]

        vocals_data, mixture_data = np.load(file=audio_files[0]), np.load(file=audio_files[1])

        return vocals_data, mixture_data, self.folders[idx].name


class Musdb19HQ_per5sec(Dataset):
    def __init__(self, split='train'):
        training_path = Path(os.path.join(os.getcwd(), f"musdb18hq/{split}"))
        self.vocals_data = np.load(file=f"{training_path}/vocals.npy")
        self.mixture_data = np.load(file=f"{training_path}/mixture.npy")
        # print(self.vocals_data.shape)
    def __len__(self):
        return self.vocals_data.shape[0]

    def __getitem__(self, idx):
        # print(torch.Tensor(self.vocals_data[idx]).unsqueeze(0).shape)
        # add channel
        return torch.Tensor(self.vocals_data[idx]).unsqueeze(0), torch.Tensor(self.mixture_data[idx]).unsqueeze(0), 'None'

class InferenceDataset(Dataset):
    def __init__(self):
        self.training_path = Path(os.path.join(os.getcwd(), f"musdb18hq/inference"))
        self.folders = [f for f in self.training_path.iterdir() if f.is_dir()]
        # print(len(self.folders))
    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        audio_files = f"{self.training_path}/{self.folders[idx].name}/mixture.npy"

        mixture_data = np.load(file=audio_files)

        return mixture_data, self.folders[idx].name


if __name__ == '__main__':
    # mix_file()

    get_npy(split='train')
    get_npy(split='val')
    get_npy(split='test')

    # get_npy_inference()

    # get_full_npy_per_5sec(split='train')
    # get_full_npy_per_5sec(split='val')
    # get_full_npy_per_5sec(split='test')

    # move_to_validation()



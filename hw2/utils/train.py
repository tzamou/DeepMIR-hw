import torch
import torch.nn as nn
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio
from torchmetrics.audio import SignalDistortionRatio
from utils.models import Separator
from utils.transforms import TorchSTFT, ComplexNorm

class SDRLoss(nn.Module):
    def __init__(self):
        super(SDRLoss, self).__init__()
        self.sdr = SignalDistortionRatio()

    def forward(self, s_true, s_pred):
        sdr_value = self.sdr(s_pred, s_true)
        return -torch.mean(sdr_value)

class TrainerOnWav:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size=16, zeroworker=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Used device is {self.device}.')

        self.model = model.to(device=self.device)

        # self.model = OpenUnmix(nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=512).to(self.device)
        # self.model.load_state_dict(torch.load(r'C:\Users\Neil Lee\Documents\Daniel\hw2\npy_stft_torch\model_epoch_50.pth', map_location=self.device))

        if zeroworker:
            num_workers = 0
        else:
            num_workers = os.cpu_count() // 2

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.loss_func = nn.MSELoss()
        # self.loss_func = SDRLoss()


        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def train(self, lr, weight_decay, lr_decay_gamma, lr_decay_patience, epochs=150, save_wav=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_decay_gamma,
            patience=lr_decay_patience,
            cooldown=10)

        loss_lst = []
        sdr_lst = []
        val_loss_lst = []
        val_sdr_lst = []
        current_lr_lst = []
        os.makedirs(f'./result/{self.date}/')
        t0 = time.time()

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            batch_t0 = time.time()

            for i, (vocals_data, mixture_data, _) in enumerate(self.train_loader):
                self.model.train()
                vocals_data = torch.tensor(vocals_data.to(device=self.device), dtype=torch.float)
                mixture_data = torch.tensor(mixture_data.to(device=self.device), dtype=torch.float)
                optimizer.zero_grad()
                try:
                    sep_vocals_data = self.model(mixture_data)[:, 0, :, :]
                except:
                    print(mixture_data)
                    raise Exception
                train_loss = self.loss_func(sep_vocals_data, vocals_data)

                train_loss.backward()
                optimizer.step()

                loss_lst.append(train_loss.item())
                # sdr_lst.append(acc)
                current_lr = optimizer.param_groups[0]['lr']
                current_lr_lst.append(current_lr)

                if i % 10 == 0 or i == len(self.train_loader):
                    print(f'\rEpoch [{epoch + 1}/{epochs}], Iteration [{i}/{len(self.train_loader)}], Learning Rate: {current_lr:.7f}, Train Loss: {train_loss.item():.4f} used time: {time.time()-batch_t0:.4f}.', end='')
                    batch_t0 = time.time()
                torch.cuda.empty_cache()
            print()
            if ((epoch+1)==1 or (epoch+1)==25 or (epoch+1)%50==0):
                torch.save(self.model, f'./result/{self.date}/{epoch+1}checkpoints.pth')
                val_loss, val_sdr = self.evaluate(split='val', save_info=True, save_wav=save_wav, name=f'epoch-{epoch+1}-')
                self.evaluate(split='test', save_info=True, save_wav=save_wav, name=f'epoch-{epoch + 1}-')
                val_sdr_lst.append(val_sdr)
                val_loss_lst.append(val_loss)
                scheduler.step(val_loss)


        training_time = time.time() - t0
        torch.save(self.model, f'./result/{self.date}/model.pth')

        np.save(file=f'./result/{self.date}/loss.npy', arr=np.array(loss_lst))
        np.save(file=f'./result/{self.date}/val_loss.npy', arr=np.array(val_loss_lst))
        np.save(file=f'./result/{self.date}/sdr.npy', arr=np.array(sdr_lst))
        np.save(file=f'./result/{self.date}/val_sdr.npy', arr=np.array(val_sdr_lst))
        np.save(file=f'./result/{self.date}/current lr.npy', arr=np.array(current_lr_lst))
        self.plot_training_loss(loss=loss_lst)
        with open(f'./result/{self.date}/result.txt', 'a') as fp:
            fp.write(f"\nThe result after training\n")
            fp.write(f"Training time: {training_time:.2f}s.\n")

    def evaluate(self, split='val', save_info=True, save_wav=True, name=''):
        assert split in ['val', 'test']
        if split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        self.model.eval()
        test_loss = 0
        sdr_lst = []

        with torch.no_grad():
            t0 = time.time()
            SDR = SignalDistortionRatio().to(device=self.device)
            for i, (vocals_data, mixture_data, song_name) in tqdm(enumerate(loader), total=len(loader)):  # 1, 1, 1280, t
                vocals_data = torch.tensor(vocals_data, dtype=torch.float).to(device=self.device)
                mixture_data = torch.tensor(mixture_data, dtype=torch.float).to(device=self.device)
                sep_vocals_data = self.model(mixture_data)[:, 0, :, :]
                test_loss += self.loss_func(sep_vocals_data, vocals_data).item()

                # estimated_sources = sep_vocals_data.squeeze(0, 1).cpu().numpy()
                # reference_sources = vocals_data.squeeze(0, 1).cpu().numpy()

                if len(sep_vocals_data.squeeze(0, 1).shape)==1:
                    sdr = SDR(sep_vocals_data.squeeze(0, 1), vocals_data.squeeze(0, 1))
                    sdr_lst.append(sdr.item())
                else:
                    for i in range(sep_vocals_data.squeeze(0, 1).shape[0]):
                        sdr = SDR(sep_vocals_data.squeeze(0, 1)[i], vocals_data.squeeze(0, 1)[i])
                        sdr_lst.append(sdr.item())

                if save_wav:
                    if not os.path.exists(f'./result/separable/{name}/'):
                        os.makedirs(f'./result/separable/{name}/')
                    # print(sep_vocals_data.shape)
                    torchaudio.save(fr"./result/separable/{name}/{song_name[0]}.wav", sep_vocals_data.squeeze(0).cpu(), 44100)
                    print('save!')

            evaluate_time = time.time()-t0

        # acc = np.mean(np.array(acc))
        sdr_lst = np.array([np.array(sublist) for sublist in sdr_lst])
        test_loss /= len(self.test_loader.dataset)
        test_sdr = np.median(np.array(sdr_lst).flatten())

        if save_info:
            if not os.path.exists(f'./result/{self.date}/'):
                os.makedirs(f'./result/{self.date}/')
            np.save(file=f'./result/{self.date}/testing-sdr.npy', arr=np.array(sdr_lst))
            with open(f'./result/{self.date}/result.txt', 'a') as fp:
                fp.write(f"Model name: {self.model.model_name}.\n")
                fp.write(f"Info: {name}, Data split: {split}\n")
                fp.write(f"Evaluate time: {evaluate_time:.2f}s.\n")
                fp.write(f"Testing loss is {test_loss:.5f}.\n")
                fp.write(f"Testing SDR is {test_sdr:.5f}.\n")

        print(f'The evaluate time: {evaluate_time:.2f}s, test loss is {test_loss:.5f}, SDR is {test_sdr:.5f}.')

        return test_loss, test_sdr

    def inference(self, dataset):
        num_workers = os.cpu_count() // 2
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.model.eval()
        with torch.no_grad():
            t0 = time.time()
            for i, (mixture_data, song_name) in tqdm(enumerate(loader)): #1, 1, 1280, t
                mixture_data = torch.tensor(mixture_data, dtype=torch.float).to(device=self.device)
                sep_vocals_data = self.model(mixture_data)[:, 0, :, :]

                if not os.path.exists(f'./result/separable/inference/'):
                    os.makedirs(f'./result/separable/inference/')
                torchaudio.save(fr"./result/separable/inference/{song_name[0]}.wav", sep_vocals_data.squeeze(0).cpu(), 44100)
                print('save!')
            inference_time = time.time()-t0
            print(f'Inference time is {inference_time:.2f} s.')


    def plot_training_loss(self, loss: list):
        plt.title('Training Loss', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Loss Value', fontsize=15)
        plt.plot(loss, label='train')
        plt.grid(True)
        plt.savefig(f'./result/{self.date}/loss.png')
        plt.clf()
        # plt.show()



class TrainerOnSpectrogram:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size=16, zeroworker=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Used device is {self.device}.')

        self.model = model.to(device=self.device)
        self.encoder = nn.Sequential(TorchSTFT(n_fft=4096, n_hop=1024, center=True), ComplexNorm(mono=True)).to(device=self.device)

        # self.model = OpenUnmix(nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=512).to(self.device)
        # self.model.load_state_dict(torch.load(r'C:\Users\Neil Lee\Documents\Daniel\hw2\npy_stft_torch\model_epoch_50.pth', map_location=self.device))
        if zeroworker:
            num_workers = 0
        else:
            num_workers = os.cpu_count() // 2

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.loss_func = nn.MSELoss()

        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def train(self, lr, weight_decay, lr_decay_gamma, lr_decay_patience, epochs=150, save_wav=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_decay_gamma,
            patience=lr_decay_patience,
            cooldown=10)

        loss_lst = []
        sdr_lst = []
        val_loss_lst = []
        val_sdr_lst = []
        current_lr_lst = []
        os.makedirs(f'./result/{self.date}/')
        t0 = time.time()

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(epochs):
            batch_t0 = time.time()
            for i, (vocals_data, mixture_data, _) in enumerate(self.train_loader):
                self.model.train()
                vocals_data = torch.tensor(vocals_data.to(device=self.device), dtype=torch.float)
                mixture_data = torch.tensor(mixture_data.to(device=self.device), dtype=torch.float)
                optimizer.zero_grad()

                stft_mixture_data = self.encoder(mixture_data)
                # print(stft_mixture_data.shape)
                sep_stft_vocals_data = self.model(stft_mixture_data)  # [:, 0, :, :]
                stft_vocals_data = self.encoder(vocals_data)
                # print(sep_stft_vocals_data.shape)

                train_loss = self.loss_func(sep_stft_vocals_data, stft_vocals_data)

                train_loss.backward()
                optimizer.step()

                loss_lst.append(train_loss.item())
                # sdr_lst.append(acc)
                current_lr = optimizer.param_groups[0]['lr']
                current_lr_lst.append(current_lr)

                if i % 10 == 0 or i == len(self.train_loader):
                    print(f'\rEpoch [{epoch + 1}/{epochs}], Iteration [{i}/{len(self.train_loader)}], Learning Rate: {current_lr:.7f}, Train Loss: {train_loss.item():.4f} used time: {time.time()-batch_t0:.4f}.', end='')
                    batch_t0 = time.time()
                torch.cuda.empty_cache()
            print()
            if ((epoch+1)==1 or (epoch+1)==25 or (epoch+1)==50 or (epoch+1)==150 or (epoch+1)==200 or (epoch+1)==250):
                torch.save(self.model, f'./result/{self.date}/{epoch+1}checkpoints.pth')
                val_loss, val_sdr = self.evaluate(split='val', save_info=True, save_wav=save_wav, name=f'epoch-{epoch+1}-')
                val_sdr_lst.append(val_sdr)
                val_loss_lst.append(val_loss)
                scheduler.step(val_loss)

        training_time = time.time() - t0
        torch.save(self.model, f'./result/{self.date}/model.pth')

        np.save(file=f'./result/{self.date}/loss.npy', arr=np.array(loss_lst))
        np.save(file=f'./result/{self.date}/val_loss.npy', arr=np.array(val_loss_lst))
        np.save(file=f'./result/{self.date}/sdr.npy', arr=np.array(sdr_lst))
        np.save(file=f'./result/{self.date}/val_sdr.npy', arr=np.array(val_sdr_lst))
        np.save(file=f'./result/{self.date}/current lr.npy', arr=np.array(current_lr_lst))
        self.plot_training_loss(loss=loss_lst)
        with open(f'./result/{self.date}/result.txt', 'a') as fp:
            fp.write(f"\nThe result after training\n")
            fp.write(f"Training time: {training_time:.2f}s.\n")

    def evaluate(self, split='val', save_info=True, save_wav=True, name=''):
        separator = Separator(target_models={'vocals': self.model}, nb_channels=1, reconstructed='default').to(device=self.device)
        assert split in ['val', 'test']
        if split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        self.model.eval()
        separator.eval()
        test_loss = 0
        sdr_lst = []
        sdr_lst2 = []

        with torch.no_grad():
            t0 = time.time()
            SDR = SignalDistortionRatio().to(device=self.device)
            for i, (vocals_data, mixture_data, song_name) in tqdm(enumerate(loader), total=len(loader)):  # 1, 1, 1280, t
                vocals_data = torch.tensor(vocals_data, dtype=torch.float).to(device=self.device)
                mixture_data = torch.tensor(mixture_data, dtype=torch.float).to(device=self.device)


                sep_vocals_data = separator(mixture_data)[:, 0, :, :]

                nonvocal_data = mixture_data-vocals_data
                sep_nonvocal = mixture_data-sep_vocals_data

                # raise Exception

                # test_loss += self.loss_func(sep_vocals_data, vocals_data).item()

                # estimated_sources = sep_vocals_data.squeeze(0, 1).cpu().numpy()
                # reference_sources = vocals_data.squeeze(0, 1).cpu().numpy()

                if len(sep_vocals_data.squeeze(0, 1).shape)==1:
                    sdr = SDR(sep_vocals_data.squeeze(0, 1), vocals_data.squeeze(0, 1))
                    sdr_lst.append(sdr.item())
                    sdr2 = SDR(sep_nonvocal.squeeze(0, 1), nonvocal_data.squeeze(0, 1))
                    sdr_lst2.append(sdr2.item())
                else:
                    for i in range(sep_vocals_data.squeeze(0, 1).shape[0]):
                        sdr = SDR(sep_vocals_data.squeeze(0, 1)[i], vocals_data.squeeze(0, 1)[i])
                        sdr_lst.append(sdr.item())
                        sdr2 = SDR(sep_nonvocal.squeeze(0, 1)[i], nonvocal_data.squeeze(0, 1)[i])
                        sdr_lst2.append(sdr2.item())
                if save_wav:
                    if not os.path.exists(f'./result/separable/{name}/'):
                        os.makedirs(f'./result/separable/{name}/')
                    # print(sep_vocals_data.shape)
                    torchaudio.save(fr"./result/separable/{name}/{song_name[0]}.wav", sep_nonvocal.squeeze(0).cpu(), 44100)
                    print('save!')

            evaluate_time = time.time()-t0

        # acc = np.mean(np.array(acc))
        sdr_lst = np.array([np.array(sublist) for sublist in sdr_lst])
        sdr_lst2 = np.array([np.array(sublist) for sublist in sdr_lst2])
        test_loss /= len(self.test_loader.dataset)
        test_sdr = np.median(np.array(sdr_lst).flatten())
        test_sdr2 = np.median(np.array(sdr_lst2).flatten())

        if save_info:
            if not os.path.exists(f'./result/{self.date}/'):
                os.makedirs(f'./result/{self.date}/')
            np.save(file=f'./result/{self.date}/testing-sdr.npy', arr=np.array(sdr_lst))
            np.save(file=f'./result/{self.date}/testing-nonvocal-sdr.npy', arr=np.array(sdr_lst2))
            with open(f'./result/{self.date}/result.txt', 'a') as fp:
                fp.write(f"Model name: {self.model.model_name}.\n")
                fp.write(f"Info: {name}, Data split: {split}\n")
                fp.write(f"Evaluate time: {evaluate_time:.2f}s.\n")
                fp.write(f"Testing loss is {test_loss:.5f}.\n")
                fp.write(f"Testing SDR is {test_sdr:.5f}.\n")
                fp.write(f"Testing nonvocal SDR is {test_sdr2:.5f}.\n")
        print(f'The evaluate time: {evaluate_time:.2f}s, test loss is {test_loss:.5f}, SDR is {test_sdr:.5f}, nonvocal SDR is: {test_sdr2:.5f}')

        return test_loss, test_sdr

    def inference(self, dataset):
        num_workers = os.cpu_count() // 2
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)
        separator = Separator(target_models={'vocals': self.model}, nb_channels=1, reconstructed='default').to(device=self.device)
        self.model.eval()
        with torch.no_grad():
            t0 = time.time()
            for i, (mixture_data, song_name) in tqdm(enumerate(loader)): #1, 1, 1280, t
                mixture_data = torch.tensor(mixture_data, dtype=torch.float).to(device=self.device)
                sep_vocals_data = separator(mixture_data)[:, 0, :, :]

                if not os.path.exists(f'./result/separable/inference/'):
                    os.makedirs(f'./result/separable/inference/')
                torchaudio.save(fr"./result/separable/inference/{song_name[0]}.wav", sep_vocals_data.squeeze(0).cpu(), 44100)
                print('save!')
            inference_time = time.time()-t0
            print(f'Inference time is {inference_time:.2f} s.')


    def plot_training_loss(self, loss: list):
        plt.title('Training Loss', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Loss Value', fontsize=15)
        plt.plot(loss, label='train')
        plt.grid(True)
        plt.savefig(f'./result/{self.date}/loss.png')
        plt.clf()
        # plt.show()


import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import numpy as np
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.loss import AsymmetricLoss

def calculate_multilabel_accuracy(preds, labels, threshold=0.5):
    # The shape of preds and labels are [batch_size, num_classes].
    preds = torch.sigmoid(preds)
    predicted_classes = (preds > threshold).float()
    correct_predictions = (predicted_classes == labels).sum().item()

    accuracy = correct_predictions / labels.numel()
    return accuracy


class DLModelTraining:
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Used device is {self.device}.')
        self.model = model.to(device=self.device)

        self.loss_func = nn.BCEWithLogitsLoss()
        # self.loss_func = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0.0, disable_torch_grad_focal_loss=True)

        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=4, shuffle=True)

        self.val_dataset = val_dataset
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=True)

        self.test_dataset = test_dataset
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=True)

    def train_model(self, epochs: int, lr: float):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # scheduler = CosineAnnealingLR(optimizer, T_max=3*3749, eta_min=0)

        self.model.train()
        loss_lst = []
        acc_lst = []
        val_loss_lst = []
        val_acc_lst = []
        current_lr_lst = []
        os.makedirs(f'./result/{self.date}/')
        t0 = time.time()

        for epoch in range(epochs):
            for i, (data, label) in enumerate(self.train_loader, start=1):
                data = torch.tensor(data.to(device=self.device), dtype=torch.float)
                label = torch.tensor(label.to(device=self.device), dtype=torch.float)
                optimizer.zero_grad()

                output = self.model(data)
                train_loss = self.loss_func(output, label)
                train_loss.backward()
                optimizer.step()
                # scheduler.step()
                acc = calculate_multilabel_accuracy(output, label)

                loss_lst.append(train_loss.item())
                acc_lst.append(acc)
                current_lr = optimizer.param_groups[0]['lr']
                current_lr_lst.append(current_lr)

                if i % 10 == 0 or i == len(self.train_loader):
                    print(f'\rEpoch [{epoch+1}/{epochs}], Iteration [{i}/{len(self.train_loader)}], Learning Rate: {current_lr:.7f}, Train Loss: {train_loss.item():.4f}, Acc: {acc:.4f}%.', end='')

                if i % 300 == 0:
                    torch.cuda.empty_cache()
                    val_loss, val_acc = self.evaluate(split='val')
                    torch.save(self.model.classifier, f'./result/{self.date}/model_checkpoint_{val_acc*100:.0f}.pth')
                    val_loss_lst.append(val_loss)
                    val_acc_lst.append(val_acc)
                torch.cuda.empty_cache()
            print()
        training_time = time.time()-t0
        torch.save(self.model.classifier, f'./result/{self.date}/model.pth')
        self.model.encoder_model.save_pretrained(f'./result/{self.date}/mert')

        np.save(file=f'./result/{self.date}/loss.npy', arr=np.array(loss_lst))
        np.save(file=f'./result/{self.date}/val_loss.npy', arr=np.array(val_loss_lst))
        np.save(file=f'./result/{self.date}/acc.npy', arr=np.array(acc_lst))
        np.save(file=f'./result/{self.date}/val_acc.npy', arr=np.array(val_acc_lst))
        np.save(file=f'./result/{self.date}/current lr.npy', arr=np.array(current_lr_lst))
        self.plot_training_loss(loss=loss_lst)
        self.plot_training_accuracy(acc=acc_lst)
        with open(f'./result/{self.date}/result.txt', 'a') as fp:
            fp.write(f"\nThe result after training\n")
            fp.write(f"Training time: {training_time:.2f}s.\n")


    def evaluate(self, split='val', save=True):
        assert split in ['val', 'test']
        if split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        self.model.eval()
        test_loss = 0
        acc = []
        all_out = []
        all_labels = []
        with torch.no_grad():
            t0 = time.time()
            for i, (data, label) in tqdm(enumerate(loader)):
                data = torch.tensor(data, dtype=torch.float).to(device=self.device)
                label = torch.tensor(label, dtype=torch.float).to(device=self.device)

                output = self.model(data)

                acc.append(calculate_multilabel_accuracy(output, label))
                test_loss += self.loss_func(output, label).item()
                y_pred = (output > 0.5).int()#.cpu().numpy()
                label = label.int()#.cpu().numpy()
                all_out.append(y_pred)
                all_labels.append(label)


            evaluate_time = time.time()-t0

        acc = np.mean(np.array(acc))
        test_loss /= len(self.test_loader.dataset)

        all_out = torch.cat(all_out, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        categories = ['Piano', 'Percussion', 'Organ', 'Guitar', 'Bass',
                      'Strings', 'Voice', 'Wind Instruments', 'Synth']
        if not save:
            print(f'The accuracy is {acc*100:.4f}%, evaluate time: {evaluate_time:.2f}s.')
            print(classification_report(all_labels, all_out, target_names=categories))
        else:
            with open(f'./result/{self.date}/classification_report.txt', 'a') as fp:
                report_str = classification_report(all_labels, all_out, target_names=categories)  # 獲取文字格式報告
                fp.write(report_str)
                fp.write('\n')

            with open(f'./result/{self.date}/result.txt', 'a') as fp:
                fp.write(f"Model name: {self.model.model_name}.\n")
                fp.write(f"Evaluate time: {evaluate_time:.2f}s.\n")
                fp.write(f"Testing loss is {test_loss:.5f}.\n")
                fp.write(f"Testing accuracy: {acc*100:.4f}%\n")

            print(f'Test loss is {test_loss:.5f}, Accuracy is {acc*100}%.')

        return test_loss, acc


    def plot_training_loss(self, loss: list):
        plt.title('Training Loss', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Loss Value', fontsize=15)
        plt.plot(loss, label='train')
        plt.grid(True)
        plt.savefig(f'./result/{self.date}/loss.png')
        plt.clf()
        # plt.show()

    def plot_training_accuracy(self, acc: list):
        plt.title('Training Accuracy', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Accuracy (%)', fontsize=15)
        plt.plot(acc, label='train')
        plt.grid(True)
        plt.savefig(f'./result/{self.date}/acc.png')
        plt.clf()
        # plt.show()
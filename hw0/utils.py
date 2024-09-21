import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import numpy as np

def plot_confusion_matrix(pred, labels, folder=None, DL=True, plot=False):
    pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(labels, pred)

    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix', fontsize=17)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(cm.shape[1]),
                yticklabels=range(cm.shape[0]),
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=15)
    plt.ylabel('True Label', fontsize=15)
    if plot:
        plt.plot()
    else:
        if DL:
            plt.savefig(f'./result/DL/{folder}/confusion matrix.png')
        else:
            plt.savefig(f'./result/ML/{folder}/confusion matrix.png')
        plt.clf()

def top_3_accuracy(pred, labels):
    top_3_preds = np.argsort(pred, axis=1)[:, -3:]
    correct = np.any(top_3_preds == labels[:, None], axis=1)
    top_3_acc = np.mean(correct)
    return top_3_acc



class CustomLRScheduler:
    # transformer原汁原味
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class LinearLRScheduler:
    def __init__(self, optimizer, start_lr, end_lr, total_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        # 計算當前學習率
        lr = self.start_lr + (self.end_lr - self.start_lr) * (self.step_num / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class DLModelTraining:
    def __init__(self, model, train_dataset, test_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Used device is {self.device}.')
        self.model = model.to(device=self.device)
        # self.loss_func = nn.NLLLoss()
        weight = train_dataset.get_categorical_weight().to(device=self.device)
        self.loss_func = nn.CrossEntropyLoss(weight=weight)

        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=256, shuffle=True)

        self.test_dataset = test_dataset
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=64, shuffle=True)

    def train_model(self,
                    epochs: int,
                    lr: float):
        """
        訓練CNN模型。
        :param epochs: 訓練次數
        :param lr: 學習率
        :param save_path: 模型儲存的路徑
        :return: None
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # scheduler = CustomLRScheduler(optimizer, d_model=992, warmup_steps=3000)
        # scheduler = LinearLRScheduler(optimizer, 1e-4, 3e-4, 94*25)

        self.model.train()
        loss_lst = []
        acc_lst = []
        current_lr_lst = []
        os.makedirs(f'./result/DL/{self.date}/')
        t0 = time.time()

        for epoch in range(epochs):
            for i, (data, label) in enumerate(self.train_loader, start=1):

                data = torch.tensor(data.to(device=self.device), dtype=torch.float)
                label = torch.tensor(label.to(device=self.device))
                optimizer.zero_grad()

                output = self.model(data)
                train_loss = self.loss_func(output, label)
                train_loss.backward()
                optimizer.step()
                # scheduler.step()

                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.view_as(label).eq(label).sum().item()
                acc = correct/len(label)*100

                loss_lst.append(train_loss.item())
                acc_lst.append(acc)
                current_lr = optimizer.param_groups[0]['lr']
                current_lr_lst.append(current_lr)

                if i % 10 == 0 or i == len(self.train_loader):
                    print(f'\rEpoch [{epoch+1}/{epochs}], Iteration [{i}/{len(self.train_loader)}], Learning Rate: {current_lr:.7f}, Train Loss: {train_loss.item():.4f}, Acc: {acc:.4f}%.', end='')
                if i % 100 == 0:
                    self.evaluate()
            print()
        training_time = time.time()-t0
        torch.save(self.model, f'./result/DL/{self.date}/model.pth')
        np.save(file=f'./result/DL/{self.date}/loss.npy', arr=np.array(loss_lst))
        np.save(file=f'./result/DL/{self.date}/acc.npy', arr=np.array(acc_lst))
        np.save(file=f'./result/DL/{self.date}/current lr.npy', arr=np.array(current_lr_lst))
        self.plot_training_loss(loss=loss_lst)
        self.plot_training_accuracy(acc=acc_lst)
        with open(f'./result/DL/{self.date}/result.txt', 'a') as fp:
            fp.write(f"\nThe result after training\n")
            fp.write(f"Training time: {training_time:.2f}s.\n")


    def evaluate(self, save=True):
        """
        用測試資料評估模型的準確率跟計算誤差。
        :return: None
        """

        self.model.eval()
        test_loss = 0
        correct = 0
        all_out = []
        all_labels = []
        with torch.no_grad():
            t0 = time.time()
            for i, (data, label) in tqdm(enumerate(self.test_loader)):
                data = torch.tensor(data, dtype=torch.float).to(device=self.device)
                label = torch.tensor(label).to(device=self.device)

                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.view_as(label).eq(label).sum().item()
                all_out.append(output)
                all_labels.append(label)

                test_loss += self.loss_func(output, label).item()
            evaluate_time = time.time()-t0
        acc = correct / len(self.test_loader.dataset)
        test_loss /= len(self.test_loader.dataset)

        all_out = torch.cat(all_out, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        plot_confusion_matrix(all_out.cpu().numpy(), all_labels.cpu().numpy(), folder=self.date, plot=not save)
        top3acc = top_3_accuracy(all_out.cpu().numpy(), all_labels.cpu().numpy())
        if not save:
            print(f'The accuracy is {acc*100:.4f}%, top 3 accuracy is {top3acc*100:.4f}%, evaluate time: {evaluate_time:.2f}s.')
        else:
            with open(f'./result/DL/{self.date}/result.txt', 'a') as fp:
                fp.write(f"Model name: {self.model.model_name}.\n")
                fp.write(f"Evaluate time: {evaluate_time:.2f}s.\n")
                fp.write(f"Testing loss is {test_loss:.5f}.\n")
                fp.write(f"Testing accuracy: {acc * 100:.4f}%\n")
                fp.write(f"Testing top 3 accuracy: {top3acc * 100:.4f}%\n")
            print(f'Test loss is {test_loss:.5f}, Accuracy is {acc * 100}%, Top 3 Accuracy is {top3acc * 100}%.')


    def plot_training_loss(self,
                           loss: list):
        """
        繪製訓練損失圖。
        :param loss: 訓練損失變化
        :return: None
        """

        plt.title('Training Loss', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Loss Value', fontsize=15)
        plt.plot(loss)
        plt.grid(True)
        plt.savefig(f'./result/DL/{self.date}/loss.png')
        plt.clf()
        # plt.show()

    def plot_training_accuracy(self, acc):
        """
        繪製訓練準確率圖。
        :param acc: 訓練準確率變化
        :return: None
        """
        plt.title('Training Accuracy', fontsize=17)
        plt.xlabel('Training Steps', fontsize=15)
        plt.ylabel('Accuracy (%)', fontsize=15)
        plt.plot(acc)
        plt.grid(True)
        plt.savefig(f'./result/DL/{self.date}/acc.png')
        plt.clf()
        # plt.show()
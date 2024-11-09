import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

def plot_training_sdr(metrics, model_name, metrics_name, realmean):
    plt.figure(figsize=(10, 8))
    plt.title(f'Distribution of {metrics_name} for generation data', fontsize=20)
    # plt.xlabel('Models', fontsize=17)
    plt.ylabel(metrics_name, fontsize=17)

    plt.hlines(y=realmean, xmin=0.9, xmax=9, colors='green', label=f'average {metrics_name} in real', linestyles='--', linewidth=1.5)
    plt.boxplot(metrics, showmeans=True, meanprops={'marker': 'o'}, labels=model_name)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', fontsize=15)
    plt.show()

csv_files = glob.glob('./*/*.csv')
csv_files.append(r'E:\python code\NTU_GICE\DLMIR\hw3\tutorial\Pop1K7\midi_analyzed\pop1k7.csv')
# print(csv_files)
H1_lst = []
H4_lst = []
GS_lst = []
model_name_lst = ['ckp150\nt=0.3\ntopk=1', 'ckp150\nt=0.3\ntopk=5', 'ckp150\nt=1.2\ntopk=1', 'ckp150\nt=1.2\ntopk=1',
                  'ckp200\nt=1.2\ntopk=1', 'ckp200\nt=1.2\ntopk=5', 'ckp50\nt=0.3\ntopk=1', 'ckp50\nt=0.3\ntopk=1',
                  '\nReal Data']
for csv_file in csv_files:
    folder = csv_file.split('\\')[1]
    df = pd.read_csv(csv_file).values
    H1 = df[:, 1]
    H4 = df[:, 2]
    GS = df[:, 3]
    H1_lst.append(H1)
    H4_lst.append(H4)
    GS_lst.append(GS)
    # with open('./result.txt', 'a') as f:
    #     f.write(folder+'\n')
    #     f.write(f'Mean H1: {np.mean(H1):.4f}, Median H1: {np.median(H1):.4f}, Max H1: {np.max(H1):.4f}, Min H1: {np.min(H1):.4f}, Std H1: {np.std(H1):.4f}\n')
    #     f.write(f'Mean H4: {np.mean(H4):.4f}, Median H4: {np.median(H4):.4f}, Max H4: {np.max(H4):.4f}, Min H4: {np.min(H4):.4f}, Std H4: {np.std(H4):.4f}\n')
    #     f.write(f'Mean GS: {np.mean(GS):.4f}, Median GS: {np.median(GS):.4f}, Max GS: {np.max(GS):.4f}, Min GS: {np.min(GS):.4f}, Std GS: {np.std(GS):.4f}\n')
    #     f.write('--------------\n')

plot_training_sdr(H1_lst, model_name_lst, 'H1', np.mean(H1_lst[-1]))
plot_training_sdr(H4_lst, model_name_lst, 'H4', np.mean(H4_lst[-1]))
plot_training_sdr(GS_lst, model_name_lst, 'GS', np.mean(GS_lst[-1]))


import numpy as np
import matplotlib.pyplot as plt

loss = np.load('GPT-2/training_loss.npy')
plt.title('Training Loss', fontsize=17)
plt.xlabel('Training Steps', fontsize=15)
plt.ylabel('Loss Value', fontsize=15)
plt.plot(loss, label='train')
plt.grid(True)
plt.savefig(f'./GPT-2/training_loss.png')
plt.clf()
# plt.show()
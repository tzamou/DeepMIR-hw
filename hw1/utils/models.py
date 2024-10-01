import numpy as np
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier_net = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=(5, ), stride=(3, )),
                                            nn.Dropout1d(p=0.1),
                                            nn.GELU(),
                                            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=(5, ), stride=(3, )),
                                            nn.Dropout1d(p=0.1),
                                            nn.GELU(),
                                            nn.Flatten(),
                                            nn.Linear(in_features=1280, out_features=9))

        self.model_name = 'CNN-classifier'

    def forward(self, x):
        x = self.classifier_net(x)

        return x


class MERTCNNModel(nn.Module):
    def __init__(self, encoder_freeze=True, classifier_weight=None):
        super(MERTCNNModel, self).__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.encoder_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

        if encoder_freeze:
            for param in self.encoder_model.parameters():
                param.requires_grad = False


        if classifier_weight is None:
            self.classifier = Classifier()
        else:
            self.classifier = torch.load(classifier_weight)
            for param in self.classifier.parameters():
                param.requires_grad = False
        self.model_name = 'MERT-CNN'

    def forward(self, x):
        x = self.processor(x, sampling_rate=24000, return_tensors="pt")
        x = self.encoder_model(x['input_values'].squeeze(0).cuda(), x['attention_mask'].reshape(-1, 1).cuda(), output_hidden_states=True)
        x = x.hidden_states[-1].squeeze()  # torch.Size([374, 1024])
        if len(x.shape)==2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    audio = np.load('../slakh/train/Track00001_22.npy')
    audio = torch.tensor(audio).reshape((1, -1))
    audio = torch.repeat_interleave(audio, 2).reshape((2, -1))

    model = MERTCNNModel()
    model.to(device='cuda')
    model(audio)


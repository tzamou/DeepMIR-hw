import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(num_features=1),
                                 nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2)),
                                 nn.BatchNorm2d(num_features=16),
                                 nn.ReLU(),
                                 nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
                                 nn.BatchNorm2d(num_features=32),
                                 nn.ReLU())
    def forward(self, x):
        x = self.net(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_layers, nhead):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class CnnTrainsformerClassifier(nn.Module):
    def __init__(self, cnn, transformer_t, transformer_f):
        super(CnnTrainsformerClassifier, self).__init__()
        self.cnn = cnn
        self.transformer_t = transformer_t
        self.transformer_f = transformer_f

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=41664, out_features=64),  # 992*42
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=11))
        self.model_name = 'CnnTrainsformerClassifier'

    def forward(self, x):
        x = self.cnn(x)
        # x = self.freq_attention(x);print(x.shape)
        x_f = x.permute([0, 2, 1, 3])
        x_f = x_f.reshape(x_f.size(0), x_f.size(1), -1)
        x_f = self.transformer_f(x_f)  # ;print(x.shape)
        x_f = torch.flatten(x_f, start_dim=1)
        x_f = self.classifier(x_f)

        x = x.permute([0, 3, 1, 2])  # ;print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)#;print(x.shape)  # (batch_size, seq_len, feature_dim)
        x = self.transformer_t(x)#;print(x.shape)
        x = torch.flatten(x, start_dim=1)#;print(x.shape)
        x = self.classifier(x)  # ;print(x.shape)

        # fx = torch.flatten(fx,start_dim=1)#;print(fx.shape)
        # x = torch.cat([x, fx], dim=1)#;print(x.shape)
        x = (x+x_f)/2#;print(x.shape)
        return x

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=(shape, shape), stride=(stride, stride), padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class ShortChunkCNN(nn.Module):
    def __init__(self, n_channels=128):
        super(ShortChunkCNN, self).__init__()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer4 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer5 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer6 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer7 = Conv_2d(n_channels * 2, n_channels * 4, pooling=2)

        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, 11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.model_name = 'Short_Chunk_CNN'

    def forward(self, x):
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(3)
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


if __name__ == '__main__':
    # model = ShortChunkCNN()
    # model = CNNFeatureExtractorV2()
    model = CnnTrainsformerClassifier(cnn=CNNFeatureExtractor(),
                                      transformer_t=TransformerModel(feature_dim=512, num_layers=3, nhead=4),
                                      transformer_f=TransformerModel(feature_dim=512, num_layers=3, nhead=4))
    # 測試模型的shape變化
    x = torch.zeros(size=(64, 1, 128, 173))
    y = model.forward(x)
    print(y.shape)
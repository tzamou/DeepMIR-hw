import torch
from utils import DLModelTraining
from data import NsynthDataset
from models.DLmodel import CnnTrainsformerClassifier, CNNFeatureExtractor, TransformerModel, ShortChunkCNN

if __name__ == '__main__':
    # model = ShortChunkCNN()
    # model = CnnTrainsformerClassifier(cnn=CNNFeatureExtractor(),
    #                                   transformer_t=TransformerModel(feature_dim=992, num_layers=3, nhead=4),
    #                                   transformer_f=TransformerModel(feature_dim=1344, num_layers=3, nhead=4))

    # train_dataset = NsynthNpyDataset(datafolder='D:/Dataset/nsynth-subtrain')
    # test_dataset = NsynthNpyDataset(datafolder='D:/Dataset/nsynth-test')
    train_dataset = NsynthDataset(datafolder='E:/Dataset/nsynth-subtrain', n_fft=1024, win_length=256, hop_length=1024, use_log=True)
    test_dataset = NsynthDataset(datafolder='E:/Dataset/nsynth-test', n_fft=1024, win_length=256, hop_length=1024, use_log=True)

    model = torch.load('./result/DL/Transformer with taking the log/model.pth')
    trainer = DLModelTraining(model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    # trainer.train_model(epochs=3, lr=2e-4)
    trainer.evaluate(save=False)



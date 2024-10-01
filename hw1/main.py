import torch
from utils import DLModelTraining
from utils.data import SlakhDataset
from utils.models import MERTCNNModel

if __name__ == '__main__':
    model = MERTCNNModel(encoder_freeze=True, classifier_weight='./result/MERT-CNN/model.pth')

    # './slakh' should be replaced with the actual path to the Slakh folder on your computer.
    train_dataset = SlakhDataset(datasplit='train', folder='./slakh')
    val_dataset = SlakhDataset(datasplit='validation', folder='./slakh')
    test_dataset = SlakhDataset(datasplit='test', folder='./slakh')

    trainer = DLModelTraining(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    # trainer.train_model(epochs=1, lr=0.0005)
    trainer.evaluate(split='test', save=False)



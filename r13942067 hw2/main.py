import torch

from utils.data import Musdb19HQ, InferenceDataset, Musdb19HQ_per5sec
from utils.models import OpenUnmix, Separator
from utils.train import TrainerOnWav, TrainerOnSpectrogram

def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"Layer: {name} | Parameters: {param_count}")
            total_params += param_count
    print(f"\nTotal trainable parameters: {total_params}")

if __name__ == '__main__':
    ### The model before training
    ## waveform output
    # target_model = OpenUnmix(nb_bins=2049, nb_channels=1, nb_layers=3)
    # model = Separator(target_models={'vocals': target_model}, nb_channels=1, reconstructed='default')
    ## spectrogram output
    # model = OpenUnmix(nb_bins=2049, nb_channels=1, nb_layers=1, used_frfn=True)

    ### The model after training
    model = torch.load(r'./result/lstm3-best model.pth')
    print_model_parameters(model)

    train_dataset = Musdb19HQ(split='train')
    val_dataset = Musdb19HQ(split='val')
    test_dataset = Musdb19HQ(split='test')

    # train_dataset = Musdb19HQ_per5sec(split='train')
    # val_dataset = Musdb19HQ_per5sec(split='val')
    # test_dataset = Musdb19HQ_per5sec(split='test')

    # trainer = TrainerOnWav(model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=1, zeroworker=False)  # waveform output
    trainer = TrainerOnSpectrogram(model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, batch_size=1, zeroworker=False)  # spectrogram output
    # trainer.train(lr=0.001, lr_decay_patience=80, lr_decay_gamma=0.3, weight_decay=1e-5, epochs=300, save_wav=False)
    trainer.evaluate(split="test", save_info=True, save_wav=False, name='testing')

    # inference_dataset = InferenceDataset()
    # trainer.inference(dataset=inference_dataset)
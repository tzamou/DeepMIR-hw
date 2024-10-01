import matplotlib.pyplot as plt
from glob import glob
import os
import json
import pretty_midi
import numpy as np
import argparse

import librosa
import torch
from utils.models import MERTCNNModel
from sklearn.metrics import classification_report
import torch.nn as nn
from utils import calculate_multilabel_accuracy

with open('./class_idx2MIDIClass.json') as f:
    class_idx2MIDIClass = json.load(f)
with open('./idx2instrument_class.json') as f:
    idx2instrument_class = json.load(f)
with open('./MIDIClassName2class_idx.json') as f:
    MIDIClassName2class_idx = json.load(f)

categories = [
    'Piano', 'Percussion', 'Organ', 'Guitar', 'Bass', 
    'Strings', 'Voice', 'Wind Instruments', 'Synth'
    ]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_track_dir', type=str,
                        help='source(test) midi track folder path', default='./test_track')
    parser.add_argument('--save_dir', type=str,
                        help='saved fig folder path', default='./')
    args = parser.parse_args()
    return args

def extract_pianoroll_from_midi(midi_file_path, time_step=5.0):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    # Determine total duration in seconds
    total_time = midi_data.get_end_time()
    
    # Create an empty pianoroll matrix without the "Empty" class
    num_classes = len(class_idx2MIDIClass)
    num_time_steps = int(np.ceil(total_time / time_step))
    pianoroll = np.zeros((num_classes, num_time_steps))
    
    # Process each instrument in the MIDI file
    for instrument in midi_data.instruments:
        program_num = instrument.program
        
        if instrument.is_drum:
            instrument_class = 128
        else:
            # Determine the class for this instrument
            instrument_class = idx2instrument_class.get(str(program_num), None)
        if instrument_class and instrument_class in MIDIClassName2class_idx:
            class_idx = MIDIClassName2class_idx[instrument_class]
            
            # Fill the pianoroll for each note
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                start_idx = int(np.floor(start_time / time_step))
                end_idx = int(np.ceil(end_time / time_step))
                pianoroll[class_idx, start_idx:end_idx] = 1  # Mark the note as present

    return pianoroll

def pianoroll_comparison(true_pianoroll, pred_pianoroll, save_path):
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plotting the true pianoroll
    axes[0].imshow(true_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[0].set_title('True Labels')
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories)
    axes[0].set_xlabel('Time Steps')

    # Plotting the predicted pianoroll
    axes[1].imshow(pred_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[1].set_title('Predicted Labels')
    axes[1].set_yticks(range(len(categories)))
    axes[1].set_yticklabels(categories)
    axes[1].set_xlabel('Time Steps')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)

def main(opt):
    midi_path_list = glob(os.path.join(opt.test_track_dir, '*.mid'))
    audio_path_list = glob(os.path.join(opt.test_track_dir, '*.flac'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MERTCNNModel(classifier_weight='./result/MERT-CNN/model.pth')
    model.to(device=device)

    for src_path in midi_path_list:
        waveform, sample_rate = librosa.load(src_path.replace('.mid', '.flac'), sr=24000)
        numseg = int(np.floor(len(waveform) / 120000))
        waveform = waveform[:numseg * 120000].reshape((-1, 120000))
        pred_pianoroll = None
        t_logits = None
        for input_waveform in waveform:
            input_waveform = torch.Tensor(input_waveform).reshape(1, -1)
            input_waveform = input_waveform.to(device=device)
            output = model(input_waveform)
            # output = (nn.functional.sigmoid(output) > 0.5).int()i
            if pred_pianoroll is None:
                t_logits = output
                output = (nn.functional.sigmoid(output) > 0.5).int()
                pred_pianoroll = output
            else:
                t_logits = torch.cat([t_logits, output], dim=0)
                t = t_logits.shape[0]
                t_importance = torch.arange(t-1, -1, -1).view(t, 1).float()
                t_importance = torch.exp(-t_importance).repeat(1, 9)
                t_importance = t_importance.to(device=device)
                t_logits.to(device=device)
                output = t_logits * t_importance
                output = torch.sum(output, dim=0).unsqueeze(0)
                output = (nn.functional.sigmoid(output) > 0.5).int()
                pred_pianoroll = torch.cat([pred_pianoroll, output], dim=0)

        pred_pianoroll = pred_pianoroll.permute(1, 0)
        # ---------------------------
        name = src_path.split('/')[-1].split('.')[0]
        true_pianoroll = extract_pianoroll_from_midi(src_path)
        # pred_pianoroll is your model predict result please load your results here
        # pred_pianoroll.shape should be [9, L] and the L should be equal to true_pianoroll
        pred_pianoroll = pred_pianoroll.cpu().numpy()

        # print(pred_pianoroll.shape)
        # print(true_pianoroll.shape)
        print(calculate_multilabel_accuracy(preds=torch.Tensor(pred_pianoroll).permute(1, 0), labels=torch.Tensor(true_pianoroll).permute(1, 0)))
        pianoroll_comparison(true_pianoroll, pred_pianoroll, name+'.png')

        song_name = os.path.splitext(os.path.basename(src_path))[0]
        with open(f'./test_track/{song_name}_classification_report.txt', 'w') as fp:
            report_str = classification_report(true_pianoroll.reshape(-1, 9), pred_pianoroll.reshape(-1, 9), target_names=categories)  # 獲取文字格式報告
            fp.write(report_str)
            fp.write('\n')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
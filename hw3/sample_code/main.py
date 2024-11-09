import datetime
import time

import torch
import glob
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
import pickle
import utils
import os
import argparse
import warnings

warnings.filterwarnings('ignore')
from GPT2RGA import *

## set the input length. must be same with the model config
X_LEN = 1024


def parse_opt():
    date_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser()
    ####################################################
    # you can define your arguments here. there is a example below.
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    parser.add_argument('--ckp_folder', type=str, help='ckp folder.', default=f'./result/{date_folder}')
    ####################################################
    parser.add_argument('--dict_path', type=str, help='the dictionary path.',
                        default=r'E:\python code\NTU_GICE\DLMIR\hw3\tutorial\basic_event_dictionary.pkl')
    args = parser.parse_args()
    return args


opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))


class NewsDataset(Dataset):
    def __init__(self, midi_l=[]):
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
        # self.parser = np.load('../Pop1K7/all_src.npy')
        # np.save('../Pop1K7/all_src.npy', arr=self.parser)

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):

        return self.parser[index]

    def chord_extract(self, midi_path, max_time):
        ####################################################
        # add your chord extraction method here if you want
        ####################################################
        return

    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end

        # if you use chord items, you need to add chord_items into "items"
        # e.g. items = tempo_items + note_items + chord_items
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def prepare_data(self, midi_paths, prompt=False):
        # extract events
        all_events = []
        for path in midi_paths:
            events = self.extract_events(path)
            all_events.append(events)
        # event to word
        all_words = []

        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                    # print(self.event2word.keys())
                else:
                    # OOV
                    if event.name == 'Note Velocity':
                        # replace with max velocity based on our training data
                        words.append(self.event2word['Note Velocity_21'])
                    else:
                        # something is wrong
                        # you should handle it for your own purpose
                        print('something is wrong! {}'.format(e))
            all_words.append(words)
        if prompt:
            return all_words
        # all_words is a list containing words list of all midi files
        # all_words = [[tokens of midi], [tokens of midi], ...]

        # you can cut the data into what you want to feed into model
        # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words) - self.x_len - 1, self.x_len):
                x = words[i:i + self.x_len]
                y = words[i + 1:i + self.x_len + 1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0:len(pairs) - (len(pairs) % 5)]
            segments = segments + pairs
        segments = np.array(segments)
        print(segments.shape)
        return segments


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #################################################
        # create your model here
        #################################################
        config = GPTConfig(VOCAB_SIZE,
                           max_seq,
                           dim_feedforward=dim_feedforward,
                           n_layer=12,
                           n_head=8,
                           n_embd=512,
                           enable_rpr=True,
                           er_len=max_seq)
        self.model = GPT(config)#.to(get_device())

    def forward(self, x):
        #################################################
        # create your model here
        #################################################
        return self.model(x)[0]


def temperature_sampling(logits, temperature, topk):
    #################################################
    # 1. adjust softmax with the temperature parameter
    # 2. choose top-k highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the top-k highest probs as wav result by the probs after normalize
    #################################################
    output = logits / temperature
    probs = F.softmax(output, dim=0)

    topk_probs, topk_indices = torch.topk(probs, topk)

    topk_probs_normalized = topk_probs / topk_probs.sum()

    result_index = torch.multinomial(topk_probs_normalized, 1).item()
    final_class = topk_indices[result_index].cpu().detach().numpy()
    return final_class


def test(n_target_bar=32, temperature=1.2, topk=5, output_path='', model_path='', prompt=False, prompt_path='', init_generated_bar=0):
    # check path folder
    try:
        os.makedirs('./testing results', exist_ok=True)
        print("dir \'./testing results\' is created")
    except:
        pass

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        # load model
        checkpoint = torch.load(model_path)

        model = Model().to(opt.device)


        model.load_state_dict(checkpoint['model'])
        model.eval()

        batch_size = 1

        if prompt:
            # If prompt, load prompt file, extract events, create tokens. (similar to dataset preparation)
            data = NewsDataset()
            words = data.prepare_data(midi_paths=[prompt_path], prompt=True)
            print(f'all words length: {len(words[0])}')
            for w in words[0]:
                if 'Tempo Value' in word2event[int(w)]:
                    tv = int(w)
        else:
            # Or, random select prompt to start
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                if 'chord' in opt.dict_path:
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in event2word.items() if 'Chord' in k]
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    # test run this
                    # print('no chord!!!', event2word.items())
                    tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                    # print(tempo_values)
                    ws.append(event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        print(f'Original_length: {original_length}')
        initial_flag = 1
        current_generated_bar = init_generated_bar
        print('Start generating')
        while current_generated_bar < n_target_bar:
            print(f"\rcurrent_generated_bar: {current_generated_bar}, Seq_length: {len(words[0])}", end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])

            temp_x = torch.Tensor(temp_x).long()  # current_generated_bar: 4torch.Size([1, 2036])
            output_logits = model(temp_x.to(opt.device)[:, -max_seq:])

            # sampling
            _logit = output_logits[0, -1]  # .to('cpu').detach().numpy()
            word = temperature_sampling(logits=_logit, temperature=temperature, topk=topk)
            if 'Tempo Value' in word2event[int(word)]:
                words[0].append(tv)
            else:
                words[0].append(word)
            # words[0].append(word)
            if word == event2word['Bar_None']:
                current_generated_bar += 1


        utils.write_midi(
            words=words[0],
            word2event=word2event,
            output_path=output_path,
            prompt_path=None)

def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"Layer: {name} | Parameters: {param_count}")
            total_params += param_count
    print(f"\nTotal trainable parameters: {total_params}")

def train(is_continue=False, checkpoints_path=''):
    epochs = 200
    # create data list
    # use glob to get all midi file path
    train_list = glob.glob('../../Pop1K7/midi_analyzed/*/*.mid')
    # print(train_list)
    print('train list len =', len(train_list))

    # dataset
    train_dataset = NewsDataset(train_list)
    # dataloader
    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Dataloader is created')

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(opt.device)
    else:
        device = torch.device("cpu")

    # create model
    if not is_continue:
        start_epoch = 1
        model = Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
        else:
            os._exit()
        start_epoch = checkpoint['epoch'] + 1

        model = Model().to(device)
        model.load_state_dict(checkpoint['model'])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Model is created \nStart training')

    model.train()
    losses = []
    try:
        os.makedirs(opt.ckp_folder, exist_ok=True)
        print("dir is created")
    except:
        pass

    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        single_epoch = []
        for i in tqdm(train_dataloader):
            # x, y shape = (batch_size, length)
            x = i[:, 0, :].to(device).long()  # x torch.Size([4, 1024])
            y = i[:, 1, :].to(device).long()

            output_logit = model(x)

            loss = nn.CrossEntropyLoss()(output_logit.permute(0, 2, 1), y)
            loss.backward()
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            optimizer.zero_grad()
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, losses[-1]))
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': losses[-1],
                    }, os.path.join(opt.ckp_folder, 'epoch_%03d.pkl' % epoch))
        np.save(os.path.join(opt.ckp_folder, 'training_loss'), np.array(losses))
    print(f'Training time is {time.time() - start_time: .4f} sec.')


def main():
    ## Train GPT-2
    # train()

    ## Generate 20 songs
    model = 200
    t = 1.2
    topk = 5
    save_path = f'testing results/GPT-12-{model} t-{t} topk-{topk}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 1
    while i < 21:
        try:
            print(f'\n--------------------\nGenerate {i} song. model is {model}, t is {t}, topk is {topk}')
            test(n_target_bar=32, temperature=t, topk=topk, output_path=f'{save_path}/testing song {i}.mid', model_path=f'./result/GPT-2/epoch_{model:03d}.pkl', prompt=False)
            i += 1
        except Exception as e:
            print(e)
            continue

    ## Task 2
    # i = 1
    # while i < 4:
    #     try:
    #         temperature = 1.2
    #         topk = 5
    #         if not os.path.exists(f'testing results/prompt/GPT-12-150 t-{temperature} topk-{topk}/'):
    #             os.makedirs(f'testing results/prompt/GPT-12-150 t-{temperature} topk-{topk}/')
    #         print(f'Generate temperature: {temperature}, topk: {topk}, i: {i}.')
    #         test(n_target_bar=32, temperature=temperature, topk=topk, output_path=f'testing results/prompt/GPT-12-150 t-{temperature} topk-{topk}/song_{i}.mid',
    #              model_path=f'result/GPT-2/epoch_150.pkl', prompt=True, prompt_path=f'../prompt_song/song_{i}.mid', init_generated_bar=8)
    #         i += 1
    #         print('Save!!!!!!!!!!!!!!')
    #     except Exception as e:
    #             print(e)
    #             continue


if __name__ == '__main__':
    main()

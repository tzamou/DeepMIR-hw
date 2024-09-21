# Homework-0 Musical note classification

### r13942067 陳冠霖

Code cloud drive link:

Github link:

# How to run the code

## Task 1: Visualize a Mel-Spectrogram
You can used the **data.py** to plot and save the mel-spectrogram. the arguments 'datafolder' is the folder that include wav file.

I used the nsynth-subtrain's wav file to plot the spectrogram.
```python
plot_mel_spec(datafolder='E:/Dataset/nsynth-subtrain/audio/')
```

## Task 2: Traditional ML Model
### Convert data to npy file.
**In data.py, used wav2npy() to read all wav files and convert to npy file. the arguments 'datafolder' is the dataset folder.**

'datafolder' recommend used the nsynth-subtrain, nsynth-train and nsynth-test path. If save as npy file, data.npy and label.npy will exist in “datafolder”/npy/ 

E.g. if run wav2npy() and datafolder=“E:/Dataset/nsynth-subtrain”, data.npy will exist in E:/Dataset/nsynth-subtrain/npy/data.npy
```python
wav2npy(datafolder='E:/Dataset/nsynth-subtrain', sr=44100,  n_fft=1024, hop_length=1024)
wav2npy(datafolder='E:/Dataset/nsynth-test', sr=44100,  n_fft=1024, hop_length=1024)
```

### Training the ML model
**Run ML_train.py will evaluate the random forest classification model. Before training ML_model should convert wav file to npy, run data.py will convert to npy.**

ML_train.py is used in training ML model. evaluate_model() can evaluate the model after training.
training_ML() and train a RandomForest-Classifier.

The arguments 'model_path' is the after training model path; testing_folder is the path of nsynth-test; training_folder is the path of nsynth-subtrain
(need add the /npy/ in path like 'E:/Dataset/nsynth-subtrain/npy/')
training data's shape is (48037, 128, 173)
testing data's shape is (4096, 128, 173)
```python

training_ML(training_folder=training_folder, testing_folder=testing_folder)
evaluate_model(model_path=model_path, testing_folder=testing_folder)
```

## Task 3: Deep Learning Model
The main.py can training the DL model and evaluate the model. 

**Run the main.py will evaluate the model.**

train_dataset and test_dataset are the dataset, the arguments "datafolder" is the dataset path, "use_log" let Mel-spectrograms with or without taking the log.

```python
train_dataset = NsynthDataset(datafolder='E:/Dataset/nsynth-subtrain', n_fft=1024, win_length=256, hop_length=1024, use_log=True)
test_dataset = NsynthDataset(datafolder='E:/Dataset/nsynth-test', n_fft=1024, win_length=256, hop_length=1024, use_log=True)
```
This model is after training, the model without taking the log is in "Transformer **without** taking the log"
```python
model = torch.load('./result/DL/Transformer with taking the log/model.pth')
```
Used this code can evaluate the model.
```python
trainer = DLModelTraining(model=model, train_dataset=train_dataset, test_dataset=test_dataset)
trainer.evaluate(save=False)
```

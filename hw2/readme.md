# Data pre-processing

### Please change all training_path entries in data.py to the path of musdb18hq on your computer!

You can convert the wav files in the dataset to npy files and save them by running *utils/data.py*.

The *move_to_validation()* method can move the validation data from the train folder to an automatically created validation folder.

The *mix_file()* function can mix other non-vocal data together for calculating the SDR of non-vocal.

The *get_full_npy_per_5sec()* function can split the dataset into many five-second segments and save them as a single npy file.

In the musdb18hq folder, create a new folder with the song's name and add the song's wav file inside (the song name should be changed to **mixture.wav**). At this point, you can use *get_npy_inference()* to convert these songs to npy format for inference.

For example, if you want to add a song by AC/DC (I absolutely love this band), you can create a path like "musdb18hq/inference/AC DC - Highway to Hell/mixture.wav" to store the song and use *get_npy_inference()* to convert it to .npy format.

# Training and inferencing

In main.py, you can choose which model to use, including models with waveform outputs and spectrogram outputs. Then, you can use print_model_parameters(model) to print the number of parameters in the model. The trainer.evaluate function can evaluate the model on the validation dataset (split='val') or the testing dataset (split='test').

If you want to perform inference on data outside the dataset, please use the following code to load the data and perform inference.

```python
inference_dataset = InferenceDataset()
trainer.inference(dataset=inference_dataset)
```
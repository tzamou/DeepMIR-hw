# Homework-1 Instrument activity detection

### r13942067 陳冠霖

Code cloud drive link: https://drive.google.com/drive/folders/1Mi_fRQGMJqvnAFQssX-UFaVxX-sZ2KC8?usp=sharing

Github link:

# How to run the code

## Inference the testing data
You can run main.py to perform inference on the test dataset. After execution, the accuracy and classification report will be printed in the terminal.

When using SlakhDataset, make sure to set the folder parameter to the path of the dataset. You only need to specify the path to the Slakh dataset.

```python
# './slakh' should be replaced with the actual path to the Slakh folder on your computer.
train_dataset = SlakhDataset(datasplit='train', folder='./slakh')
```
After run the code, you can get the result.
```
The accuracy is 83.5682%, evaluate time: 69.78s.
                  precision    recall  f1-score   support

           Piano       0.92      0.89      0.90      1889
      Percussion       0.89      0.27      0.41       243
           Organ       0.65      0.33      0.43       461
          Guitar       0.91      0.96      0.93      1943
            Bass       0.97      0.97      0.97      2076
         Strings       0.92      0.67      0.78      1235
           Voice       0.60      0.61      0.61       485
Wind Instruments       0.80      0.23      0.36       889
           Synth       0.90      0.26      0.41       647

       micro avg       0.90      0.74      0.81      9868
       macro avg       0.84      0.58      0.64      9868
    weighted avg       0.89      0.74      0.78      9868
     samples avg       0.88      0.74      0.78      9868
```
## Inference the test track
For the inference results on the test track, you can run plot_pianoroll.py.

If you do not want to calculate the weighted sum of each time step, the simplest approach is to comment out lines **115** to **122** in the code, which corresponds to the following lines.
```python
t_logits = torch.cat([t_logits, output], dim=0)
t = t_logits.shape[0]
t_importance = torch.arange(t-1, -1, -1).view(t, 1).float()
t_importance = torch.exp(-t_importance).repeat(1, 9)
t_importance = t_importance.to(device=device)
t_logits.to(device=device)
output = t_logits * t_importance
output = torch.sum(output, dim=0).unsqueeze(0)
```

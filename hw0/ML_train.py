from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from utils import top_3_accuracy, plot_confusion_matrix
import os
import datetime
import time
import joblib

def training_ML(training_folder:str, testing_folder:str):
    '''

    :param training_folder: the training data's folder path. e.g. "E:/Dataset/nsynth-subtrain/npy/"
    :param testing_folder: the testing data's folder path. e.g. "E:/Dataset/nsynth-test/npy/"
    :return:
    '''
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(f'./result/ML/{date}/')

    model = RandomForestClassifier(n_estimators=100,)
    # joblib.dump(model, f'./result/ML/{date}/model_param.joblib')

    training_data = np.load(f'{training_folder}data.npy')
    training_label = np.load(f'{training_folder}label.npy')
    training_data = training_data.reshape((training_data.shape[0], -1))

    print("training...")
    t0 = time.time()
    model.fit(training_data, training_label)
    fitting_time = time.time() - t0
    joblib.dump(model, f'./result/ML/{date}/random_forest_model.pkl')

    testing_data = np.load(f'{testing_folder}data.npy')
    testing_label = np.load(f'{testing_folder}label.npy')
    testing_data = testing_data.reshape((testing_data.shape[0], -1))

    print("evaluating...")
    pred = model.predict(testing_data)
    pred_proba = model.predict_proba(testing_data)
    acc = accuracy_score(testing_label, pred)
    top3acc = top_3_accuracy(pred_proba, testing_label)
    with open(f'./result/ML/{date}/result.txt', 'a') as fp:
        fp.write(f"ML algorithm is: {model.__class__.__name__}.\n")
        fp.write(f"Training time: {fitting_time:.2f}s.\n")
        fp.write(f"Accuracy: {acc*100:.4f}%\n")
        fp.write(f"Top 3 accuracy: {top3acc*100:.4f}%\n")
    plot_confusion_matrix(pred_proba, testing_label, folder=date, DL=False)

def evaluate_model(model_path: str, testing_folder: str):
    '''

    :param model_path: the .pkl file of ML model path.
    :param testing_folder: the testing data's folder path. e.g. "E:/Dataset/nsynth-test/npy/".
    :return: print top 1 & 3 accuracy and plot confusion matrix.
    '''

    model = joblib.load(model_path)
    testing_data = np.load(f'{testing_folder}data.npy')
    testing_label = np.load(f'{testing_folder}label.npy')
    testing_data = testing_data.reshape((testing_data.shape[0], -1))
    pred_proba = model.predict_proba(testing_data)
    pred = model.predict(testing_data)

    acc = accuracy_score(testing_label, pred)
    top3acc = top_3_accuracy(pred_proba, testing_label)
    print(f'The accuracy is {acc*100:.4f}%, top 3 accuracy is {top3acc*100:.4f}%.')
    plot_confusion_matrix(pred_proba, testing_label, DL=False, plot=True)

if __name__ == '__main__':
    model_path = './result/ML/ML result/random_forest_model.pkl'
    testing_folder = 'E:/Dataset/nsynth-test/npy/'
    training_folder = 'E:/Dataset/nsynth-subtrain/npy/'

    # training_ML(training_folder=training_folder, testing_folder=testing_folder)
    evaluate_model(model_path=model_path, testing_folder=testing_folder)
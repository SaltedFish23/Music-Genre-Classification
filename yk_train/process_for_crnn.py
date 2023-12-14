import os

import librosa
import numpy as np
import sklearn.model_selection as skms
import torch


def compute_melgram(audio_path):
    '''
    return a mel-spectrogram in shape of (96, 1366)
    '''

    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) // 2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS), ref=np.max)
    return ret

def data_read(directory=r"..\..\dataset\archive\Data\genres_original"):
    classes = {'blues': 0,
               'classical': 1,
               'country': 2,
               'disco': 3,
               'hiphop': 4,
               'jazz': 5,
               'metal': 6,
               'pop': 7,
               'reggae': 8,
               'rock': 9}

    mels = []
    labels = []

    x = 0

    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "jazz.00054.wav":  # skip the broken one
                x += 1
                continue
            filename = os.path.join(dirname, filename)
            mel = compute_melgram(filename)
            label = classes[filename.split('\\')[6]]

            mels.append(mel)
            labels.append(label)
            x += 1


    return mels, labels

class GTZANDataset_CRNN():
    def __init__(self, rootDir=r"..\..\dataset\archive\Data\genres_original"):
        mels, labels = data_read(rootDir)
        mels = np.array(mels)
        labels = np.array(labels)
        print(mels.shape)
        mels_train, mels_test, labels_train, labels_test = skms.train_test_split(mels, labels, test_size=0.3, random_state=0)
        mels_train = torch.Tensor(mels_train)
        mels_test = torch.Tensor(mels_test)
        print(mels_train.shape)
        labels_train = torch.Tensor(labels_train)
        labels_test = torch.Tensor(labels_test)

        self.trainDataset = torch.utils.data.TensorDataset(mels_train, labels_train)
        self.testDataset = torch.utils.data.TensorDataset(mels_test, labels_test)

    def __call__(self, train="False"):
        """
        :param train:
        :return: dataset
        """
        if train == "True":
            return self.trainDataset
        elif train == "False":
            return self.testDataset


if __name__ == "__main__":
    # mel = compute_melgram(r"..\..\dataset\archive\Data\genres_original\blues\blues.00000.wav")
    # print(mel)
    # print(mel.shape)

    dataset = GTZANDataset_CRNN()
    print(dataset.trainDataset[0])


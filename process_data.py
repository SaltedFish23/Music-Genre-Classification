import numpy as np
import librosa
import torch
import os
from PIL import Image
from multiprocessing import Pool
from torch.utils.data import random_split, Dataset


class PNGToMFCC(object):

    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc

    def __call__(self, img):
        # 将PIL图像转换为灰度图像
        gray_image = img.convert('L')

        # 将灰度图像转换为数字数组
        mel_spec_array = np.array(gray_image, dtype=float)

        # 将数字数组转换为梅尔频谱图
        mel_spec = librosa.feature.inverse.mel_to_stft(mel_spec_array)

        # 将梅尔频谱图转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 计算MFCC系数
        mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=self.n_mfcc)

        # 选择前几个MFCC系数作为特征
        mfcc_features = mfcc[:self.n_mfcc]
        mfcc_features = np.resize(mfcc_features,(32,448))

        # 将特征矩阵转换为PyTorch张量
        mfcc_tensor = torch.from_numpy(mfcc_features)

        return mfcc_tensor.float()
    


class my_dataset(Dataset):
    classes = {'blues': 0,
               'classical': 1,
               'country': 2,
               'disco': 3,
               'hiphop': 4,
               'jazz': 5,
               'metal':6,
               'pop': 7,
               'reggae': 8,
               'rock': 9}

    def __init__(self, root_dir): #将root_dir路径下的image文件转化为pytorch张量
        self.data = []
        self.targets = []
        images_folders = os.listdir(root_dir)
        for label_dir in images_folders:
            label = my_dataset.classes[label_dir]
            images_files = os.listdir(os.path.join(root_dir, label_dir))
            images_files_path = [os.path.join(root_dir, label_dir, file) for file in images_files]
            for image_path in images_files_path:
                image = Image.open(image_path)
                transform = PNGToMFCC()
                image_mfcc = transform(image)
                self.data.append(image_mfcc)
                self.targets.append(label)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class GTZANDataset:
    def __init__(self, rootDir=r"..//dataset//archive//Data//images_original"):
        self.rootDir = rootDir
        self.transform = PNGToMFCC()

        self.data = my_dataset(rootDir)

        self.trainDataset, self.testDataset = random_split(
            dataset=self.data,
            lengths=[int(len(self.data) * 0.7), len(self.data) - int(len(self.data) * 0.7)],
            generator=torch.Generator().manual_seed(0)
        )

    def __call__(self, train="False"):
        """
        :param train:
        :return: dataset (every picture is transformed into tensor whose size is [13, 432])
        """
        if train == "True":
            return self.trainDataset
        elif train == "False":
            return self.testDataset

    def data(self):
        return self.data


if __name__ == "__main__":
    root_dir_path = "..//data//music//images_original"
    target_dir_path = "..//data//music//processed//MusicGen2.pth"
    dataset = my_dataset(root_dir_path)
    dataset = dataset.data
    torch.save(dataset, target_dir_path)
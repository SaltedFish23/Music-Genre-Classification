import torch
from torch.utils.data import random_split


def read_data(root_dir_path):
    dataset = torch.load(root_dir_path)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[int(len(dataset) * 0.7), len(dataset) - int(len(dataset) * 0.7)],
        generator=torch.Generator().manual_seed(0)
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    data_path = ""
    train_dataset, test_dataset = read_data(data_path)

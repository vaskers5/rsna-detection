import os


import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


from lib.datasets import TitsSet


DEVICE = 'cpu'


def create_full_df():
    detailed_class = pd.read_csv('data/stage_2_detailed_class_info.csv')
    train_labels = pd.read_csv('data/stage_2_train_labels.csv')
    df = train_labels.merge(detailed_class, on='patientId')
    train_folder = 'data/stage_2_train_images/'
    df['img_path'] = df.patientId.apply(lambda item: os.path.abspath(os.path.join(train_folder, f'{item}.dcm')))
    return df


def collate_batch(batch):
    img, rectangle, label = zip(*batch)
    # rectangle = pad_sequence(rectangle, batch_first=True, padding_value=0)

    # label = pad_sequence(label, batch_first=True, padding_value=0)

    return torch.stack(img).to(DEVICE), rectangle, label



if __name__ == "__main__":
``
    df = create_full_df()
    dataset = TitsSet(df.sample(n=2000))
    full_loader = DataLoader(dataset, batch_size=10, collate_fn=collate_batch)
    for batch in full_loader:
        img, rectangle, label = batch
        print('aboba')


import pandas as pd
import pydicom

import torch
from torch.utils.data  import Dataset
from tqdm import tqdm
import torchvision.transforms as T


def get_rectangle(df_item):
    try:
        x, y, w, h = [int(item) for item in (df_item['x'], df_item['y'], df_item['width'], df_item['height'])]
    except:
        return None
    return (x, y, w, h)



class TitsSet(Dataset):
    def __init__(self,
     tits_df: pd.DataFrame,
     transforms :torch.nn.Module = T.Compose([torch.from_numpy])) -> None:
        self.transforms = transforms
        self.images, self.rectangles, self.labels = [], [], []
        for p_id in tqdm(set(tits_df.patientId.to_list())):
            p_df = tits_df[tits_df['patientId'] == p_id]

            self.rectangles.append(p_df.apply(get_rectangle, axis=1).to_list())

            self.images.append(p_df.img_path.iloc[0])
            self.labels.append(p_df.Target.to_list())
    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = pydicom.dcmread(self.images[idx]).pixel_array
        rectangle = self.rectangles[idx]
        label = self.labels[idx]
        return self.transforms(img), rectangle, label

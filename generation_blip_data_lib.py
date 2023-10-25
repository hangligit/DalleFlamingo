import json
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def load_image_list(dtype):
    if dtype=='nocaps':
        annotation_file = 'nocaps_val_4500_captions.json'
        image_folder = '/home/ubuntu/datasets/nocaps/val/'
        data = json.load(open(annotation_file, 'r'))
        df = pd.DataFrame(data['images'])
        filenames = df.file_name.values.tolist()
        filenames = [image_folder + file for file in filenames]
    else:
        if dtype=='coco_val':
            annotation_file = 'annotation/coco_karpathy_val.json'
        elif dtype=='coco_test':
            annotation_file = 'annotation/coco_karpathy_test.json'
        image_folder = '/home/ubuntu/datasets/coco/images/'
        data = json.load(open(annotation_file, 'r'))
        filenames = [image_folder + file["image"] for file in data]
    return filenames


class NocapsEval(Dataset):
    def __init__(self, dtype='nocaps', image_size=384):
        self.images = load_image_list(dtype)

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.transform = transform


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path.split('/')[-1]


def create_loader(batch_size=32, num_workers=4, dtype='nocaps',image_size=384):
    dataset=NocapsEval(dtype,image_size)
    loader=DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader

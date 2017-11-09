import os
import random
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class TripletImageDataset(Dataset):
    """
    Creates anchor, positive, negative triples from diretory of Image folders
    """

    def __init__(self, root, transform=None):
        self.name_to_id = dict((name, i)
                               for i, name in enumerate(sorted(os.listdir(root))))
        self.id_to_name = dict((i, name)
                               for name, i in self.name_to_id.items())
        self.images = glob(os.path.join(root, "*", "*"))
        self.class_dirs = glob(os.path.join(root, "*"))
        self.transform = transform

    def create_triplets(self):
        class_1, class_2 = random.sample(self.class_paths, 2)
        images_1 = os.listdir(class_1)
        image_2 = os.listdir(class_2)
        return class_1, class_2, images_1, image_2

    def __getitem__(self, idx):
        anchor_path = random.choice(self.images)
        anchor_class_dir, _ = os.path.split(anchor_path)
        positive_image_paths = glob(os.path.join(anchor_class_dir, '*'))
        positive_path = random.choice(positive_image_paths)

        while True:
            negative_class_dir = random.choice(self.class_dirs)
            if negative_class_dir != anchor_class_dir:
                break

        negative_image_paths = glob(os.path.join(negative_class_dir, '*'))
        negative_path = random.choice(negative_image_paths)

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.images)

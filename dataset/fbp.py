import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FBP(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, transform=False):
        super(FBP, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = {
            0:'unlabeled', 1:'industrial area', 2:'paddy field', 3:'irrigated field', 
            4:'dry cropland', 5:'garden land', 6:'arbor forest', 7:'shrub forest', 
            8:'park', 9:'natural meadow', 10:'artificial meadow', 11:'river', 
            12:'urban residential', 13:'lake', 14:'pond', 15:'fish pond',
            16:'snow', 17:'bareland', 18:'rural residential', 19:'stadium',
            20:'square', 21:'road', 22:'overpass', 23:'railway station', 24:'airport'
        }
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((6800, 7200))

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index][:-4]
        image = Image.open(os.path.join(self.root, "rgb_images/" + self.ids[index])) # w, h
        sample['image'] = image
        # sample['image'] = transforms.functional.adjust_contrast(image, 1.4)
        if self.label:
            # label = scipy.io.loadmat(join(self.root, 'Notification/' + self.ids[index].replace('_sat.jpg', '_mask.mat')))["label"]
            # label = Image.fromarray(label)
            label = Image.open(os.path.join(self.root, 'fbp_labels/' + self.ids[index].replace('.tif', '_24label.png')))
            sample['label'] = label
        if self.transform and self.label:
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}
        return sample

    def _transform(self, image, label):
        # if np.random.random() > 0.5:
        #     image = self.color_jitter(image)

        # if np.random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     label = transforms.functional.vflip(label)

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     degree = 60 * np.random.random() - 30
        #     image = transforms.functional.rotate(image, degree)
        #     label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     ratio = np.random.random()
        #     h = int(2448 * (ratio + 2) / 3.)
        #     w = int(2448 * (ratio + 2) / 3.)
        #     i = int(np.floor(np.random.random() * (2448 - h)))
        #     j = int(np.floor(np.random.random() * (2448 - w)))
        #     image = self.resizer(transforms.functional.crop(image, i, j, h, w))
        #     label = self.resizer(transforms.functional.crop(label, i, j, h, w))
        
        return image, label


    def __len__(self):
        return len(self.ids)
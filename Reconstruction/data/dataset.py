import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class StanfordDog(data.Dataset):
    """
    Loading StanfordDog dataset

    """
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        Initialization of the dataset
        root : place holder of the mnist dataset
        transforms : required transformation of the images
        train / test : getting training set or testing set

        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

    def __getitem__(self, index):
        pass

    

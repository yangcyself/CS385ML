import os, glob, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from xml.etree import ElementTree as ET

class StanfordDog(data.Dataset):
    """
    Loading StanfordDog dataset

    """
    def __init__(self, root, transforms=None, train=True, already=False):
        """
        Initialization of the dataset
        root : place holder of the mnist dataset
        transforms : required transformation of the images
        train / test : getting training set or testing set

        """
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor()
                ])
        if already:
            self.breed_dict = {}
            self.imgs = []
            if os.path.exists(path=root+"/dogs.pickle"):
                with open(root+"/dogs.pickle", 'rb') as load_data:
                    self.imgs, self.labels = pickle.load(load_data)
            for img, label in zip(self.imgs, self.labels):
                self.breed_dict[label] = img
        else:
            self.train = train
            self.breed_dict = {}
            self.imgs = []
            self.name = []
            self.labels = []
            annots = glob.glob(root + '/Annotation/*/*')
            print(annots[-1])

            for annot in annots:
                text = open(annot, 'r').read()
                annot_list = annot.split('/')
                rt = ET.fromstring(text)
                children = rt.getchildren()
                objects = children[5:]

                for object in objects:
                    objChildren = object.getchildren()
                    breed = objChildren[0].text

                    img = cv2.imread(root + '/Images/' + annot_list[-2] + '/' + annot_list[-1] + '.jpg')
                    self.name.append(root + '/Images/' + annot_list[-2] + '/' + annot_list[-1] + '.jpg')
                    #print("Processing " + root + '/Images/' + annot_list[-2] + '/' + annot_list[-1] + '.jpg')
                    xmin = int(objChildren[4].getchildren()[0].text)
                    xmax = int(objChildren[4].getchildren()[2].text)
                    ymin = int(objChildren[4].getchildren()[1].text)
                    ymax = int(objChildren[4].getchildren()[3].text)

                    # cv2.imshow("img", img[ymin:ymax, xmin:xmax])
                    # cv2.waitKey(0)
                    self.imgs.append(img[ymin:ymax, xmin:xmax])
                    self.labels.append(breed)
                    if breed in self.breed_dict.keys():
                        self.breed_dict[breed].append(img[ymin:ymax, xmin:xmax])
                    else:
                        self.breed_dict[breed] = [img[ymin:ymax, xmin:xmax]]

    def save(self):
        with open("./dogs.pickle", 'wb') as save_data:
            data_list = [self.imgs, self.labels]
            pickle.dump(data_list, save_data)

    def __getitem__(self, index):
        tmp = cv2.resize(self.imgs[index], (96, 96))
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(tmp)
        img = self.transforms(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
    sd = StanfordDog('.', '.')
    sd.save()

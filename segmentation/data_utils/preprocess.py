import numpy as np
import os
import cv2
from config import *

def image2label(img):
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), cv2.INTER_CUBIC)
    _, y = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)
    y = y.reshape(1, INPUT_WIDTH*INPUT_HEIGHT)
    return y

def label2image(y):
    img = y.reshape(INPUT_WIDTH, INPUT_HEIGHT)
    _, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
    return img

def process_segmentations():
    seg_path = DATA_PATH + "/segmentations/"
    image_list = open(DATA_PATH + "/images.txt").readlines()
    for image in image_list:
        image = image.split(' ')[1][:-4] + "png"
        img = cv2.imread(os.path.join(seg_path, image), cv2.IMREAD_COLOR)
        seg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(seg, 0.5, 1, cv2.THRESH_BINARY)
        img[:, :, 0] = seg
        cv2.imwrite(os.path.join(seg_path, image), img)

def move_images():
    train_test_split = DATA_PATH + "/train_test_split.txt"

    image_dir = DATA_PATH + "/images/"
    label_dir = DATA_PATH + "/segmentations/"
    train_images_dir = DATA_PATH + "/train_images/"
    val_images_dir = DATA_PATH + "/val_images/"
    train_labels_dir = DATA_PATH + "/train_labels/"
    val_labels_dir = DATA_PATH + "/val_labels/"

    image_list_file = open(DATA_PATH + "/images.txt", 'r').readlines()
    isval = open(train_test_split, 'r').readlines()
    N = len(image_list_file)

    for i in range(N):
        image_filename = image_list_file[i].split()[1]
        img = cv2.imread(image_dir + image_filename, 1)

        label_filename = image_filename[:-4] + ".png"
        seg = cv2.imread(label_dir + label_filename, 1)
        isvali = int(isval[i].split(' ')[1])
        
        image_name = image_filename.split('/')[-1]
        label_name = label_filename.split('/')[-1]
        if isvali:
            cv2.imwrite(train_images_dir + image_name, img)
            cv2.imwrite(train_labels_dir + label_name, seg)
        else:
            cv2.imwrite(val_images_dir + image_name, img)
            cv2.imwrite(val_labels_dir + label_name, seg)
            

if __name__ == "__main__":
    process_segmentations()
    move_images()


# f1 = open(images_file_path, 'r')
# img_paths = [s.split()[1] for s in f1.readlines()]
# f2 = open(train_text_split_path, 'r')
# is_training_images = [int(s.split()[1]) for s in f2.readlines()]
#
# NUM = len(is_training_images)
# print("total number of images: %d" % NUM)
# n = sum(is_training_images)
# print("number of training images: %d" % n)
# x_train = np.zeros((n, m, m, 3), dtype=np.uint8)
# y_train = np.zeros((n, m*m), dtype=np.uint8)
# x_test = np.zeros((NUM - n, m, m, 3), dtype=np.uint8)
# y_test = np.zeros((NUM - n, m*m), dtype=np.uint8)
# i_train, i_test = 0, 0
#
# for i in range(NUM):
#     if i % 1000 == 0:
#         print("processing %dth images" % i)
#     x = cv2.imread(DATA_PATH + "/images/" + img_paths[i], cv2.IMREAD_COLOR)
#     segment_path = DATA_PATH + "/segmentations/" + img_paths[i][:-4] + ".png"
#     y = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
#
#     if is_training_images[i]:
#         x_train[i_train] = cv2.resize(x, (m, m), interpolation=cv2.INTER_CUBIC)
#         y_train[i_train] = image2label(y)
#         i_train += 1
#     else:
#         x_test[i_test] = cv2.resize(x, (m, m), interpolation=cv2.INTER_CUBIC)
#         y_test[i_test] = image2label(y)
#         i_test += 1
#
# print("finish processing, saving...")
# np.savez(DATA_PATH+"/train.npz", x=x_train, y=y_train)
# np.savez(DATA_PATH+"/test.npz", x=x_test, y=y_test)



import numpy as np
import cv2

a = cv2.resize(cv2.imread('ori_dog.png'), (224, 224))
b = cv2.resize(cv2.imread('vae_gen.png'), (224, 224))
c = cv2.resize(cv2.imread('vaegan.jpeg'), (224, 224))
# d = cv2.resize(cv2.imread('difSVM1.png'), (96, 96))
# e = cv2.resize(cv2.imread('difSVM.png'), (96, 96))
# f = cv2.resize(cv2.imread('difSVM5.png'), (96, 96))

r1 = np.hstack(tup=(a,b,c))
# r2 = np.hstack(tup=(d,e,f))
# r = np.vstack((r1, r2))
cv2.imshow('r', r1)
cv2.waitKey(0)

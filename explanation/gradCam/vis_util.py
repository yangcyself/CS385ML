import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from copy import deepcopy
def BGR_RGB(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return new_img

def cvt255(img):
    img -= img.min()
    img /= img.max()
    img *= 255.0
    return np.uint8(img)

def img_out(out):
    out = out.cpu().detach().numpy().transpose(1,2,0)
    out - cvt255(out)
    return BGR_RGB(out)

def img_gradient( gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient = cvt255(gradient)
    return  BGR_RGB(gradient)
    # plt.imshow( BGR_RGB(np.uint8(gradient)))
    # plt.show()
    # cv2.imwrite(filename, np.uint8(gradient))

def img_gradcam( gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return BGR_RGB(np.uint8(gcam))
    # plt.imshow( BGR_RGB(np.uint8(gcam)))
    # plt.show()
    # cv2.imwrite(filename, np.uint8(gcam))

def img_blockedArea(img, pos, size):
    timg = deepcopy(img)
    dim = 3 if len(img[0].shape) ==3 else 2
    if(dim ==3):
        for t, p in zip(timg,pos):
            t[p[0]:p[0]+size[0],p[1]:p[1]+size[1],: ] = 0 # DOG
    else:
        for t, p in zip(timg,pos):
            t[p[0]:p[0]+size[0],p[1]:p[1]+size[1]] = 0 # DOG
    return timg
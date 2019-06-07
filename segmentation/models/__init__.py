model_from_name = {}

from . import fcn
from config import *

model_from_name["fcn_8"] = fcn.fcn_8
model_from_name["fcn_32"] = fcn.fcn_32
model_from_name["fcn_8_alexnet"] = fcn.fcn_8_alexnet
model_from_name["fcn_32_alexnet"] = fcn.fcn_32_alexnet
model_from_name["fcn_8_vgg"] = fcn.fcn_8_vgg
model_from_name["fcn_32_vgg"] = fcn.fcn_32_vgg
model_from_name["fcn_8_resnet50"] = fcn.fcn_8_resnet50
model_from_name["fcn_32_resnet50"] = fcn.fcn_32_resnet50

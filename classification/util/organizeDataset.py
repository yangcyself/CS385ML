# -^- coding:utf-8 -^-
"""
Orgainze the data folder as:
data/
    train/
        dogs1/
            dog001.jpg
            dog002.jpg
            ...
        dogs2/
            dog001.jpg
            dog002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        dogs/
            dog001.jpg
            dog002.jpg
            ...
"""
import subprocess
def run(a):
# a=["wc", "-l", "file.txt"]
    print("$" + " ".join(a))
    os.system(" ".join(a))
    # y = subprocess.check_output(a).decode("utf-8").strip()
    # print(y)

import os
import sys
TrainPercentage = 0.8
dataroot = os.path.abspath( os.path.join(os.getcwd(), "../","data"))
print("dataroot",dataroot)
imgroot = os.path.join(dataroot,"Images")
trainroot =os.path.join(dataroot,"train")
validationroot = os.path.join(dataroot,"validation")



run(["mkdir","-p",trainroot])
run(["mkdir","-p",validationroot])

for folder in os.listdir(imgroot):
    tfolder = os.path.join(trainroot,folder)
    vfolder = os.path.join(validationroot,folder)
    ifolder = os.path.join(imgroot,folder)
    run(["mkdir","-p",tfolder])
    run(["mkdir","-p",vfolder])
    # break
    pics = os.listdir(ifolder)
    for p in pics[:int(len(pics)*TrainPercentage)]:
        picpath = os.path.join(ifolder,p)
        run(["cd",tfolder , "&&" ,"cp","-s",picpath,"."])
    for p in pics[int(len(pics)*TrainPercentage):]:
        picpath = os.path.join(ifolder,p)
        run(["cd",vfolder , "&&" ,"cp","-s",picpath,"."])


    
    


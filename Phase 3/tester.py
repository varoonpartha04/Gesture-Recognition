from os import listdir
from os.path import isfile, join
mypath = "3_class_gesture_data/W"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
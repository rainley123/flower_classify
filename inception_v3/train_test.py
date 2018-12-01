import os 
import shutil

FLOWER_FILE = "/home/ley/PycharmProjects/flower_classify/flower_photos"
TRAIN_FLOWER = "./train_flower"
TEST_FLOWER = "./test_flower"
VAL_FLOWER = "./validation_flower"

flower_dir = []
for dir in os.listdir(FLOWER_FILE):
    if os.path.isdir(os.path.join(FLOWER_FILE, dir)):
        flower_dir.append(dir)

for dir in flower_dir:
    os.makedirs(os.path.join(TRAIN_FLOWER, dir))
    os.makedirs(os.path.join(TEST_FLOWER, dir))
    os.makedirs(os.path.join(VAL_FLOWER, dir))
    count = 0
    for files in os.listdir(os.path.join(FLOWER_FILE, dir)):
        if count < 120:
            shutil.copy(os.path.join(FLOWER_FILE, dir, files), os.path.join(TEST_FLOWER, dir))
            count = count + 1
            print (count, files)
        elif count < 240:
            shutil.copy(os.path.join(FLOWER_FILE, dir, files), os.path.join(VAL_FLOWER, dir))
            count = count + 1
            print (count, files)
        elif count < 600:
            shutil.copy(os.path.join(FLOWER_FILE, dir, files), os.path.join(TRAIN_FLOWER, dir))
            count = count + 1
            print (count, files)

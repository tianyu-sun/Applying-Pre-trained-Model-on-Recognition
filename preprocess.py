from skimage.transform import resize
# import pandas as pd
import numpy as np
import shutil
import cv2
import os


PATH = 'thumbnails_features_deduped_sample'
NEW_PATH = 'selected_data'
TRAIN_PATH = 'train'
TEST_PATH = 'test'


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


del_files(PATH)
if not os.access(NEW_PATH, os.F_OK):
    os.mkdir(NEW_PATH)
    os.mkdir(os.path.join(NEW_PATH, TRAIN_PATH))
    os.mkdir(os.path.join(NEW_PATH, TEST_PATH))
    
id_count = 0

for human_name in os.listdir(PATH):
    if not os.access(os.path.join(NEW_PATH, TRAIN_PATH, "{}_{}".format(id_count, human_name)), os.F_OK):
        os.mkdir(os.path.join(NEW_PATH, TRAIN_PATH, "{}_{}".format(id_count, human_name)))
        os.mkdir(os.path.join(NEW_PATH, TEST_PATH, "{}_{}".format(id_count, human_name)))
        img_count = 0
#         crop_data = pd.read_csv(os.path.join(PATH, "{}_{}".format(id_count, human_name)), 'filelist_LBP.txt'), sep="\t", header=None)
#         crop_data.columns = ["name", "x0", "x1", "x2", "x3"]
        
        for frame_name in os.listdir(os.path.join(PATH, human_name)):
            if frame_name.find(".jpg") == -1:
                continue
            if img_count < 20:
                inner_path = TRAIN_PATH
            elif img_count < 40:
                inner_path = TEST_PATH
            else:
                break
            image = np.asarray(cv2.imread(os.path.join(PATH, human_name, frame_name)))
#             frame_crop_data = crop_data.loc[crop_data.name==frame_name]
#             image = image[int(frame_crop_data.x0):int(frame_crop_data.x2), 
#                           int(frame_crop_data.x1):int(frame_crop_data.x3)]
            image_resized = resize(image, (224, 224, 3), mode='reflect')
            cv2.imwrite(os.path.join(NEW_PATH, inner_path, "{}_{}".format(id_count, human_name), 'frame_{}.jpg'.format(img_count)),
                                image_resized * 255.0)  # save the images
            img_count = img_count + 1

        id_count = id_count + 1

            
# filter images from dataset
# remove images with too much whitespace
# curate finalized dataset

from PIL import Image
import os, os.path
import numpy as np
import cv2

imgs = []
path = "./pix_data"
valid_exts = [".jpg", ".png"]
invalid_count, valid_count = 0, 0
for i, f in enumerate(os.listdir(path)):
    img_path = f"{path}/{f}"
    split_path = os.path.splitext(f)
    ext = split_path[1]
    if ext.lower() not in valid_exts: # if its not valid
        print(f"Image {i} not valid")
        os.remove() # remove from dir
        invalid_count += 1 # increment num removed
        continue
    else: # if the image is valid
        img = cv2.imread(img_path) # load image

        # use opencv to count white pixels
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pixel_count = np.sum(img_gray > 240) # count white pixels
        if pixel_count > img.shape[0] * img.shape[1] * .3: # more than 30% white
            continue

        # shrink image and save to new dir
        dh, dw =  img.shape[0], img.shape[1] // 2 # new img dimensions (shrink width)
        res = cv2.resize(img, dsize=(dh, dw), interpolation=cv2.INTER_CUBIC)   
        cv2.imwrite(f"./resized_pix_data/pix_image{str(i).zfill(4)}.png", res)
        valid_count += 1

print(f"Invalid count: {invalid_count}")
print(f"Valid count: {valid_count}")
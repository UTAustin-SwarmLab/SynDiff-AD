# Convert all images from cv2 imwrite to PIL Image
from PIL import Image
import cv2
import os
from tqdm import tqdm

experiment = 'seg'

if experiment == 'seg':
    DIR_PATH = '/store/harsh/data/bdd_synthetic_ft_ceq/img/'
    print('Converting images to PIL Image FROM:', DIR_PATH)
    for _,file in tqdm(enumerate(os.listdir(DIR_PATH)), total=len(os.listdir(DIR_PATH))):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(DIR_PATH, file))
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(DIR_PATH, file))
elif experiment == 'carla':
        DIR_PATH = '/store/harsh/carla_data_neat/synthexpert/'
        print('Converting images to PIL Image FROM:', DIR_PATH)
        
        for _,dir in tqdm(enumerate(os.listdir(DIR_PATH)), total=len(os.listdir(DIR_PATH))):
            if os.path.exists(os.path.join(DIR_PATH, dir, 'rgb_front')):
                for _,file in tqdm(enumerate(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_front'))), 
                                total=len(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_front')))):
                    if file.endswith('.png'):
                        img = cv2.imread(os.path.join(DIR_PATH, dir,'rgb_front', file))
                        pil_img = Image.fromarray(img)
                        pil_img.save(os.path.join(DIR_PATH, dir, 'rgb_front',file))
            if os.path.exists(os.path.join(DIR_PATH, dir, 'rgb_left')):
                for _,file in tqdm(enumerate(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_left'))), 
                                total=len(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_left')))):    
                    if file.endswith('.png'):
                        img = cv2.imread(os.path.join(DIR_PATH, dir,'rgb_left', file))
                        pil_img = Image.fromarray(img)
                        pil_img.save(os.path.join(DIR_PATH, dir, 'rgb_left',file))

            if os.path.exists(os.path.join(DIR_PATH, dir, 'rgb_right')):
                for _,file in tqdm(enumerate(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_right'))), 
                                total=len(os.listdir(os.path.join(DIR_PATH, dir, 'rgb_right')))):
                    if file.endswith('.png'):
                        img = cv2.imread(os.path.join(DIR_PATH, dir,'rgb_right', file))
                        pil_img = Image.fromarray(img)
                        pil_img.save(os.path.join(DIR_PATH, dir, 'rgb_right',file))

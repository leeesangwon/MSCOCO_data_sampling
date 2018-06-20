import os
import random
import shutil
import glob

from PIL import Image
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO

dataDir = './coco'
dataType = 'val2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

outDir = './coco_sample'

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())

pascal_cats = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane',
               'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'chair', 
               'dining table', 'potted plant', 'couch', 'tv']
# label name in pascal voc
#             ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 
#              'bicycle', 'boat', 'bus' ,'car', 'motorbike', 'train', 'bottle', 'chair', 
#              'dining table', 'potted plant', 'sofa', 'tv/monitor']

for cat in pascal_cats:
    out_img_folder = os.path.join(outDir, 'images', cat)
    out_gt_folder = os.path.join(outDir, 'gt', cat)

    try:
        os.makedirs(out_img_folder)
        os.makedirs(out_gt_folder)
    except FileExistsError:
        pass

    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)
    random.shuffle(imgs)
    num_saved = 0
    for img in imgs:
        min_ratio = 0.1
        img_area = img['height'] * img['width']
        min_area = img_area * min_ratio
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, areaRng=[min_area, img_area],iscrowd=None)
        anns = coco.loadAnns(annIds)
        if anns:
            random_ann = random.choice(anns)
            mask = coco.annToMask(random_ann)
            mask *= 255
            mask = Image.fromarray(mask)

            img_path = os.path.join(dataDir, 'images', dataType, img['file_name'])
            out_img_path = os.path.join(out_img_folder, img['file_name'])
            out_gt_path = os.path.join(out_gt_folder, img['file_name'].split('.')[0] + '.png')
            shutil.copy2(img_path, out_img_path)
            mask.save(out_gt_path)
            num_saved += 1
        
        if num_saved == 20:
            break

img_list = glob.glob(os.path.join(outDir, 'images/*/*'))
gt_list = glob.glob(os.path.join(outDir, 'gt/*/*'))

out_list_path = os.path.join(outDir, 'list')
try:
    os.makedirs(out_list_path)
except FileExistsError:
    pass

with open(os.path.join(out_list_path, "coco.txt"), 'w') as f:
    for i, g in zip(img_list, gt_list):
        i = i.replace(outDir + '/', '').replace('\\', '/')
        g = g.replace(outDir + '/', '').replace('\\', '/')
        print(i, g, file=f)

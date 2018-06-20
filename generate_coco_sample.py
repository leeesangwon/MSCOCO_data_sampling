import os
import random
import shutil
import glob

from PIL import Image
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO


class CocoSampler():
    def __init__(self, data_dir, data_type, out_dir, num_sample_per_category=20, min_instance_area_ratio=0.1):
        self.data_dir = data_dir
        self.data_type = data_type
        self.out_dir = out_dir
        self.out_gt_dir = os.path.join(out_dir, 'gt')
        self.out_img_dir = os.path.join(out_dir, 'images')
        self.out_list_dir = os.path.join(out_dir, 'list')
        self.num_sample_per_category = num_sample_per_category
        self.min_instance_area_ratio = min_instance_area_ratio
    
    def run(self):
        self._sample_coco()
        self._gen_list_txt() 
    
    def _sample_coco(self):
        annFile = '{}/annotations/instances_{}.json'.format(self.data_dir, self.data_type)
        coco = COCO(annFile)

        pascal_cats = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane',
                    'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'bottle', 'chair', 
                    'dining table', 'potted plant', 'couch', 'tv']
        # label name in pascal voc
        #             ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 
        #              'bicycle', 'boat', 'bus' ,'car', 'motorbike', 'train', 'bottle', 'chair', 
        #              'dining table', 'potted plant', 'sofa', 'tv/monitor']

        for cat in pascal_cats:
            out_img_folder = os.path.join(self.out_img_dir, cat)
            out_gt_folder = os.path.join(self.out_gt_dir, cat)

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
                img_area = img['height'] * img['width']
                min_area = img_area * self.min_instance_area_ratio
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, areaRng=[min_area, img_area],iscrowd=None)
                anns = coco.loadAnns(annIds)
                if anns:
                    random_ann = random.choice(anns)
                    mask = coco.annToMask(random_ann) # 0 or 1
                    mask *= 255
                    mask = Image.fromarray(mask)

                    img_path = os.path.join(self.data_dir, 'images', self.data_type, img['file_name'])
                    out_img_path = os.path.join(out_img_folder, img['file_name'])
                    out_gt_path = os.path.join(out_gt_folder, img['file_name'].split('.')[0] + '.png')
                    shutil.copy2(img_path, out_img_path)
                    mask.save(out_gt_path)
                    num_saved += 1
                
                if num_saved == self.num_sample_per_category:
                    break

    def _gen_list_txt(self):
        img_list = glob.glob(os.path.join(self.out_img_dir, '*/*'))
        gt_list = glob.glob(os.path.join(self.out_gt_dir, '*/*'))

        try:
            os.makedirs(self.out_list_dir)
        except FileExistsError:
            pass
        
        out_path = os.path.join(self.out_list_dir, "coco.txt")
        
        with open(out_path, 'w') as f:
            for i, g in zip(img_list, gt_list):
                i = i.replace(self.out_dir + '/', '').replace('\\', '/')
                g = g.replace(self.out_dir + '/', '').replace('\\', '/')
                print(i, g, file=f)


def main():    
    coco_sampler = CocoSampler(data_dir = './coco',
                               data_type = 'val2014',
                               out_dir = './coco_sample',
                               num_sample_per_category=20, 
                               min_instance_area_ratio=0.1,
                               )
    coco_sampler.run()


if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

# ann_dir = '../CocoData/annotations/instances_val2017.json'
# f = open(ann_dir)
# anns = json.load(f)

# print(anns['annotations'][1])

def find_match_img(ann, anns):
    image_id = ann['image_id']
    for img in anns['images']:
        if img['id'] == image_id:
            return img



class COCOData(Dataset):
    def __init__(self, img_dir, ann_dir, split_size=7, num_boxes=2, num_classes=90, transform=None):

        self.ann_dir = ann_dir
        f = open(ann_dir)
        anns = json.load(f)

        remove_keys = ['info', 'licenses']
        for item in remove_keys:
            del anns[item]
        
        for ann in anns['annotations']:
            matched_img = find_match_img(ann, anns)
            full_width = matched_img['width']
            full_height = matched_img['height']
            
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            ann['bbox'][0] += width/2
            ann['bbox'][1] += height/2
            
            ann['bbox'][0] /= full_width
            ann['bbox'][1] /= full_height
            ann['bbox'][2] /= full_width
            ann['bbox'][3] /= full_height
            
            remove = ['segmentation', 'area', 'iscrowd', 'id']
            for item in remove:
                del ann[item]
                
        self.img_dir = img_dir
        self.anns = anns
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.transform = transform
    
    def __len__(self):
        return len(self.anns['images'])

    def __getitem__(self, index):
        # finding image matched with index
        index_image = self.anns['images'][index]
        
        # finding image id (name of image)
        image_id = index_image['id']
        boxes = []
        
        # look through annotations, matching and adding box when annotation is for the selected image
        for ann in self.anns['annotations']:
            if ann['image_id'] == image_id:
                class_label = ann['category_id']
                x = ann['bbox'][0]
                y = ann['bbox'][1]
                width = ann['bbox'][2]
                height = ann['bbox'][3]
                boxes.append([class_label, x, y, width, height])
                
        image_file_name = index_image['file_name']
        image_path = self.img_dir + '/' + image_file_name
        image = Image.open(image_path)
        boxes = torch.tensor(boxes)
        
        # data augmentation
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        # label matrix will later become the cnn output for each cell
        label_matrix = torch.zeros(self.S, self.S, self.C + 5*self.B)
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (width * self.S, height * self.S)
            
            if label_matrix[i][j][90] == 0:
                label_matrix[i, j, 90] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 91:95] = box_coordinates
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix
from typing import Any
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as ft
from torchvision.transforms import v2
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolo
from dataset import COCOData
import utils
from loss import YoloLoss
import sys
import time

seed = 123
torch.manual_seed(seed)

'''SET SEE IMAGES TO TRUE TO SEE IMAGE WITH BBOX'''
see_images = False

# Hyperparameters
learning_rate = 0.00002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
weight_decay = 0
epochs = 3
num_workers = 3
pin_memory = True

'''SET LOAD MODEL TO TRUE TO LOAD'''
load_model = False
LOAD_MODEL_FILE = 'overfit.pth.tar'


transforms = v2.Compose([
    v2.Resize((448, 448)),
        # vvv NEW .ToTensor
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, bboxes):
        for t in transforms:
            img, bboxes = t(img), bboxes
            
        return img, bboxes
    
# Train function
def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_index, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())
    
    print(f'Mean loss was {sum(mean_loss)/len(mean_loss)}')
    

'''CHANGE IMG DIR TRAIN AND VAL TO FILE DIRECTORY'''
'''CHNAGE ANN DIR TRAIN AND VAL TO FILE DIRECTORY'''
img_dir_train = '../CocoData/train2017'
ann_dir_train = '../CocoData/annotations/instances_train2017.json'

img_dir_val = '../CocoData/val2017'
ann_dir_val = '../CocoData/annotations/instances_val2017.json'

def main():
    print("STARTING")
    model = Yolo(split_size=7, num_boxes=2, num_classes=90).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = YoloLoss()
    
    if load_model:
        utils.load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
    train_dataset = COCOData(img_dir_train, ann_dir_train, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    # val_dataset = COCOData(img_dir_val, ann_dir_val, transform=transforms)
    # val_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    
    for epoch in range(epochs):
        
        if see_images:
            for x, y in train_loader:
                x = x.to(DEVICE)
                for index in range(8):
                    bboxes = utils.cellboxes_to_boxes(model(x))
                    bboxes = utils.non_max_suppression(bboxes[index], iou_threshold=0.5, threshold=0.4)
                    utils.plot_image(x[index].permute(1,2,0).to("cpu"), bboxes)

                sys.exit()
        
        pred_boxes, target_boxes = utils.get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        
        mean_avg_precision = utils.mean_average_precision(pred_boxes, target_boxes, num_classes=90, box_format='midpoint')
        
        print(f'Train mAP {mean_avg_precision}')
        
        if mean_avg_precision > 0.9:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            utils.save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(10)
        
        train(train_loader, model, optimizer, loss_fn)
       
       
 
if __name__ == '__main__':
    main()

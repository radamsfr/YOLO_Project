import torch
import torch.nn as nn
from utils import IoU

class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=90):
        super(YoloLoss, self).__init__()
        self.mean_squared_error = nn.MSELoss(reduction='sum')
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        box1nodes = predictions[..., 91:95]
        box2nodes = predictions[..., 96:100]
        target_nodes = target[..., 91:95]
        
        # Calculating IoU of prediction box 1 and 2 to target
        # output nodes 91-95 are bounding box for box 1
        IoU_B1 = IoU(box1nodes, target_nodes)
        # output nodes 96-100 are bounding box for box 2
        IoU_B2 = IoU(box2nodes, target_nodes)
        
        # cat = concatenate = combine things
        # unsqueeze changes the shape of tensor to specified dimention
        IoUs = torch.cat([IoU_B1.unsqueeze(0), IoU_B2.unsqueeze(0)], dim=0)
        
        # best_box = 0 if box1 is best fit, = 1 if box2 is best fit
        best_box = torch.argmax(IoUs, dim=0)                # IoU_maxes, best_box = torch.max(IoUs, dim=0)
        
        # exists = 1 if object in cell exists, exists = 0 when theres no object
        exists = target[..., 90].unsqueeze(3)
        
        # loss for box coordinates
        box_predictions = exists * (best_box * box1nodes + (1-best_box) * box2nodes)
        box_targets = exists * (best_box * target_nodes)
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        box_loss = self.mean_squared_error(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_loss, end_dim=-2),
        )
        
        # loss for object exists
        pred_box = (best_box * predictions[..., 95:96] + (1-best_box) * predictions[..., 90:91])
        obj_loss = self.mean_squared_error(
            torch.flatten(exists * pred_box),
            torch.flatten(exists * target[..., 90:91])
        )
        
        # loss for object not exists
        no_obj_loss = self.mean_squared_error(
            torch.flatten((1 - exists) * predictions[..., 90:91], start_dim=1),
            torch.flatten((1 - exists) * target[..., 90:91], start_dim=1)
        )
        no_obj_loss += self.mean_squared_error(
            torch.flatten((1 - exists) * predictions[..., 95:96], start_dim=1),
            torch.flatten((1 - exists) * target[..., 90:91], start_dim=1)
        )
        
        # loss for object class
        class_loss = self.mean_squared_error(
            torch.flatten(exists * predictions[..., :90], end_dim=-2),
            torch.flatten(exists * target[..., :90], end_dim=-2)
        )
        
        # combine loss
        loss = (self.lambda_coord * box_loss) + obj_loss + (self.lambda_no_obj * no_obj_loss) + class_loss
        
        return loss
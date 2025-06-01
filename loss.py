from torch import Tensor
import torch
import torch.nn.functional as F
from torch.nn import Module
import torch.nn as nn

from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # If input image has 1 channel (grayscale), duplicate it to have 3 channels
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, focal_weight=0.5, ce_weight=0
                 , tversky_alpha=0.6, tversky_beta=0.4, tv_weight=1, focal_tv_weight=0,
                 dice_weight=0.0, delta=2/3):
        super(CombinedLoss, self).__init__()
        
        # Initialize parameters
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.delta = delta
        self.tv_weight = tv_weight
        self.focal_tv_weight = focal_tv_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def forward(self, inputs, targets, smooth=1e-6):

        # Calculate Cross-Entropy loss
        ce_loss = F.cross_entropy(inputs, targets)

        # Calculate Dice loss
        inputs_softmax = F.softmax(inputs, dim=1)[:, 1]
        targets_float = (targets == 1).float()
        intersection = (inputs_softmax * targets_float).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (inputs_softmax.sum() + targets_float.sum() + smooth))

        # Calculate Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Calculate Tversky Loss with current parameters
        true_pos = (inputs_softmax * targets_float).sum()
        false_neg = ((1 - inputs_softmax) * targets_float).sum()
        false_pos = (inputs_softmax * (1 - targets_float)).sum()
        tversky_loss = 1 - (true_pos / (true_pos + 
                            self.tversky_beta * false_neg + 
                            self.tversky_alpha * false_pos + smooth))
        
        # Calculate Focal Tversky Loss
        focal_tversky_loss = tversky_loss ** (1 / self.delta)
        self.fce =  self.focal_weight * focal_loss.mean()
        self.dice= self.dice_weight * dice_loss
        
        # Combine losses
        combined_loss = (self.dice_weight * dice_loss +
                         self.focal_weight * focal_loss +
                         self.ce_weight * ce_loss +
                         self.tv_weight * tversky_loss +
                         self.focal_tv_weight * focal_tversky_loss)

        return combined_loss
        

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

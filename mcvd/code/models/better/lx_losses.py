
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np
import torch

def compute_image_gradient(img):
    """
    img: (B, T, H, W) or (B, C, H, W)
    return: gradient magnitude: (B, T, H, W)
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    # Reshape to [B*T, 1, H, W]
    B, T, H, W = img.shape
    img = img.reshape(B * T, 1, H, W)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)

    grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)  # Avoid sqrt(0)
    grad = grad.reshape(B, T, H, W)
    return grad
    
def gradient_loss(pred, target, loss_type='l1'):
    """
    pred, target: [B, T, H, W]
    return: scalar gradient loss
    """
    pred_grad = compute_image_gradient(pred)
    target_grad = compute_image_gradient(target)

    if loss_type == 'l1':
        return F.l1_loss(pred_grad, target_grad)
    elif loss_type == 'l2':
        return F.mse_loss(pred_grad, target_grad)
    else:
        raise ValueError("Unsupported loss type. Use 'l1' or 'l2'.")
   
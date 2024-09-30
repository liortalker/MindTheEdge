# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC). All rights reserved.

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.spatial.distance import dice

kernel_size = (15,15)
kernel = torch.ones(kernel_size)
kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
kernel = nn.Parameter(data=kernel, requires_grad=False).cuda()


def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

def attention_loss2(output,target,mask=None,is_spatially_adaptive=False):

    eps = 1e-14
    if not is_spatially_adaptive:
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()
        alpha = num_neg / (num_pos + num_neg) * 1.0
    else:
        target_pos_alpha_mat = F.conv2d(target, kernel, padding=(int((kernel_size[0]-1)/2)))/(kernel_size[0]*kernel_size[1])
        target_neg_alpha_mat = 1 - target_pos_alpha_mat
        # if all neg - use 0.5
        target_neg_alpha_mat[target_neg_alpha_mat >= (1.0 - eps)] = 0.5
        alpha = target_neg_alpha_mat

    # alpha = 0.5

    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
    weight = weight.detach()

    if not mask is None:
        weight = weight * mask

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    # loss = torch.sum(loss)
    loss = torch.mean(loss)
    return loss


class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            l = label[i]
            loss_focal = attention_loss2(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss


if __name__ == '__main__':
    N = 4
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 1, H, W)).float()
    o_b = [torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W)]
    crientation = AttentionLoss2()
    total_loss = crientation(o_b, label)
    print('loss 2-1 :   '+ str(total_loss))





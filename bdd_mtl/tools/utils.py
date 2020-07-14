import numpy as np
import cv2, torch
import os
from PIL import Image
import scipy.ndimage as ndi
import boundary_utils as bu

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def normalize(a):
    a -= a.min()
    a /= a.max()
    return a

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)

def get_boundary_map(segmap):
    bitmap = np.zeros_like(segmap)
    im2, contours, hierarchy = cv2.findContours(np.asarray(segmap), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bitmap = cv2.drawContours(bitmap, contours, -1, 1, 1)
    return Image.fromarray(np.uint8(bitmap))

def distance_transform(mask, clip_background_distance=True, normalized=True):
    mask = np.asarray(mask)
    invalid = mask < 0. # {-1, 0, 1} pixel values
    foreground = mask.copy()
    background = 1. - foreground
    foreground[invalid] = 0.
    background[invalid] = 0.

    foreground_dist = ndi.distance_transform_edt(foreground)
    background_dist = ndi.distance_transform_edt(background)

    if clip_background_distance:
        foreground_max = foreground_dist.max()
        background_dist[background_dist > foreground_max] = foreground_max

    distance = foreground_dist * foreground + background_dist * background

    if normalized:
        distance = normalize(distance)

    return Image.fromarray(distance)

def db_eval_boundary(fg_boundary, gt_boundary, bound_th=0.008):
    out = [bu.db_eval_boundary(f, g, bound_th=bound_th) for f, g in zip(fg_boundary, gt_boundary)]
    return np.mean(out)

def thin_edge(pred_map):
    if len(pred_map.shape) == 2:
        return bu.thin_edge(pred_map)
    if len(pred_map.shape) == 3:
        return np.array([bu.thin_edge(p) for p in pred_map])
    if len(pred_map.shape) == 4:
        return np.array([[bu.thin_edge(q) for q in p] for p in pred_map])

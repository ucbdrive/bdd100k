import mmcv
import numpy as np
import torch
from PIL import Image, ImageEnhance

__all__ = [
    'ImageTransform', 'BboxTransform', 'MaskTransform', 'SegMapTransform',
    'Numpy2Tensor'
]


class RandomColor(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Color(image).enhance(alpha)
        return image


class RandomContrast(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Contrast(image).enhance(alpha)
        return image


class RandomSharpness(object):
    def __init__(self, var=0.4):
        self.var = var

    def __call__(self, image):
        alpha = 1.0 + np.random.uniform(-self.var, self.var)
        image = ImageEnhance.Sharpness(image).enhance(alpha)
        return image


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None,
                 color_var=0,
                 contrast_var=0,
                 sharpness_var=0):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        if color_var > 0:
            self.random_color = RandomColor(color_var)
        if contrast_var > 0:
            self.random_contrast = RandomColor(contrast_var)
        if sharpness_var > 0:
            self.random_sharpness = RandomColor(sharpness_var)

    def __call__(self, img, scale, flip=False, crop_info=None, keep_ratio=True):
        # image jittering
        try:
            img = Image.fromarray(img)
        except:
            print(img)
        if hasattr(self, 'random_color'):
            img = self.random_color(img)
        if hasattr(self, 'random_contrast'):
            img = self.random_contrast(img)
        if hasattr(self, 'random_sharpness'):
            img = self.random_sharpness(img)
        img = np.array(img)
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if crop_info is not None:
            # if crop, no need to pad
            cx1, cy1, cx2, cy2 = crop_info
            img = img[cy1:cy2, cx1:cx2]
            pad_shape = img.shape
        # pad and set pad_shape
        if crop_info is None and self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)

        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False, pad_val=0):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=pad_val) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, crop_info=None, keep_ratio=True, pad_val=0):
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)

        if crop_info is not None:
            # if crop, no need to pad
            cx1, cy1, cx2, cy2 = crop_info
            img = img[cy1:cy2, cx1:cx2]
        elif self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor, pad_val=pad_val)
        return img


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])

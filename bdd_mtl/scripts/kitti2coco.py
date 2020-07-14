import mmcv
import os
import os.path as osp
import json
import argparse
from collections import defaultdict

cats_mapping = {
    'person': 1,
    'vehicle': 2,
    'bike': 3,
}


def main():
    folder = '/home/jiangmiao/datasets/KITTI/training'
    out_file = osp.join(folder, 'kitti_train_0920.json')
    img_path = osp.join(folder, 'image_02')
    label_path = osp.join(folder, 'label_02')
    vid_names = os.listdir(img_path)
    kitti = defaultdict(list)
    for k, v in cats_mapping.items():
        kitti['categories'].append(dict(id=v, name=k))
    img_id = 0
    ann_id = 0
    for i, vid_name in enumerate(vid_names):
        video = dict(id=i, name=vid_name)
        kitti['videos'].append(video)
        img_lists = os.listdir(osp.join(img_path, vid_name))
        for img_name in img_lists:
            file_name = '{}/{}'.format(vid_name, img_name)
            img = mmcv.imread(osp.join(img_path, vid_name, img_name))
            h, w, _ = img.shape
            image = dict(
                file_name=file_name,
                height=h,
                width=w,
                id=img_id,
                video_id=i,
                index=int(img_name.split('.png')[0]))
            kitti['images'].append(image)
            img_id += 1
    mmcv.dump(kitti, out_file)


if __name__ == '__main__':
    main()

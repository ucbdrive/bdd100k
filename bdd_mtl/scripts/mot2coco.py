import mmcv
import os
import json
import argparse
from collections import defaultdict

cats_mapping = {
    'person': 1,
    'rider': 2,
    'car': 3,
    'bus': 4,
    'truck': 5,
    'bike': 6,
    'motorcycle': 7,
    # 'other': 8,
    # 'train': 9
}


def parse_args():
    parser = argparse.ArgumentParser(description='BDD Tracking to COCO format')
    parser.add_argument(
        "-d",
        "--dir",
        default="/home/jiangmiao/datasets/MOT/MOT16/",
        help="root directory of MOT images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folders = ['test', 'train']

    for folder in folders:
        coco = defaultdict(list)
        for k, v in cats_mapping.items():
            coco['categories'].append(dict(id=v, name=k))
        vid_path = os.path.join(args.dir, folder)
        vid_name = os.listdir(vid_path)
        image_id = 0
        for i, _vid_name in enumerate(vid_name):
            video = dict(id=i, name=_vid_name)
            coco['videos'].append(video)
            img_lists = os.listdir(
                os.path.join(vid_path, '{}/img1'.format(_vid_name)))
            for img_name in img_lists:
                file_name = '{}/img1/{}'.format(_vid_name, img_name)
                img = mmcv.imread(os.path.join(vid_path, file_name))
                h, w, c = img.shape
                image = dict(
                    file_name=file_name,
                    height=h,
                    width=w,
                    id=image_id,
                    video_id=i,
                    index=int(img_name.split('.jpg')[0]) - 1)
                coco['images'].append(image)
                image_id += 1
        mmcv.dump(coco,
                  os.path.join(args.dir, 'annotations/{}.json').format(folder))


if __name__ == '__main__':
    main()

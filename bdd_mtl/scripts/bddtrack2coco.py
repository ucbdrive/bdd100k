import os
import json
import argparse
from collections import defaultdict

merge_map = {
    'person': 'person',
    'rider': 'person',
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',
    'bike': 'bike',
    'motorcycle': 'bike',
    'traffic light': 'traffic light',
    'traffic sign': 'traffic sign',
    'train': 'train',
    'other': 'other'
}

cats_mapping = {
    'person': 1,
    'vehicle': 2,
    'bike': 3
    # 'car': 3,
    # 'bus': 4,
    # 'truck': 5,
    # 'bike': 6,
    # 'motorcycle': 7,
    # 'other': 8,
    # 'train': 9
}


def parse_args():
    parser = argparse.ArgumentParser(description='BDD Tracking to COCO format')
    parser.add_argument(
        "-b",
        "--bdd_dir",
        default="/path/to/bdd/label/",
        help="root directory of BDD label Json files",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        default="/save/dir",
        help="directory to save coco formatted label file",
    )
    parser.add_argument(
        "--filter_others",
        action='store_true',
        help="filter `other` category",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print('Convert BDD Tracking dataset to COCO style.')
    bdd_dir = args.bdd_dir
    save_dir = args.save_dir
    # bdd_dir = '../../labels'
    # save_dir = './coco'
    os.makedirs(save_dir, exist_ok=True)
    sets = ['train', 'val', 'test']

    video_id, image_id, ann_id, global_instance_id = 0, 0, 0, 0

    no_ann = 0

    for subset in sets:
        print('Start converting {} set...'.format(subset))

        bdd_anns_path = os.path.join(bdd_dir, subset)
        coco_outfile = os.path.join(save_dir,
                                    'bdd_tracking_{}_0918.json'.format(subset))

        video_lists = os.listdir(bdd_anns_path)
        coco = defaultdict(list)

        # categories
        for k, v in cats_mapping.items():
            coco['categories'].append(dict(id=v, name=k))

        for video_index, video_ann_file in enumerate(video_lists):
            with open(os.path.join(bdd_anns_path, video_ann_file), 'r') as f:
                video_anns = json.load(f)
            instance_id_maps = dict()

            # videos
            video = dict(id=video_id, name=video_anns[0]['video_name'])
            coco['videos'].append(video)

            # images
            for image_anns in video_anns:
                image = dict(
                    file_name=image_anns['name'].split('/')[-1],
                    height=720,
                    width=1280,
                    id=image_id,
                    video_id=video_id,
                    index=image_anns['index'])
                coco['images'].append(image)

                # annotations
                for anns in image_anns['labels']:
                    if merge_map[anns['category']] not in cats_mapping.keys():
                        continue

                    bdd_id = anns['id']
                    if bdd_id in instance_id_maps.keys():
                        instance_id = instance_id_maps[bdd_id]
                    else:
                        instance_id = global_instance_id
                        global_instance_id += 1
                        instance_id_maps[bdd_id] = instance_id

                    x1 = anns['box2d']['x1']
                    x2 = anns['box2d']['x2']
                    y1 = anns['box2d']['y1']
                    y2 = anns['box2d']['y2']
                    area = float((x2 - x1 + 1) * (y2 - y1 + 1))
                    ann = dict(
                        id=ann_id,
                        image_id=image_id,
                        category_id=cats_mapping[merge_map[anns['category']]],
                        instance_id=instance_id,
                        is_occluded=anns['attributes']['Occluded'],
                        is_truncated=anns['attributes']['Truncated'],
                        bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                        area=area,
                        iscrowd=0,
                        ignore=0,
                        segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]])

                    coco['annotations'].append(ann)
                    ann_id += 1
                if len(image_anns['labels']) == 0:
                    print(image_anns['name'])
                    no_ann += 1

                image_id += 1
            video_id += 1
        with open(coco_outfile, 'w') as f:
            json.dump(coco, f)
        print('{} set converted'.format(subset))

    print('--------Done--------')
    print('Overall: {} videos, {} images, {} annotations, {} instances'.format(
        video_id + 1, image_id + 1, ann_id + 1, global_instance_id + 1))
    print('--------------------')


if __name__ == '__main__':
    main()

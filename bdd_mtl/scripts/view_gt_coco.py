import os
import cv2
import random
from pycocotools.bdd import BDD
from collections import defaultdict


def randomcolor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (b, g, r)


def main():
    ann_file = '/Users/pangjiangmiao/Documents/jiangmiao/bdd_coco_style/bdd_tracking_val_0713.json'
    bdd = BDD(ann_file)

    # Rough unit test for video pre-process in cocoapi
    video_ids = bdd.getVideoIds()
    print('video_ids: {}'.format(video_ids))
    instance_ids = bdd.getInstanceIds()
    print('instance_ids: {}'.format(instance_ids))

    sole_img_ids = bdd.getImgIdsFromVideoId(video_ids[1])
    print('sole_img_ids: {}'.format(sole_img_ids))
    sole_instance_ids = bdd.getInstanceIdsFromVideoId(video_ids[1])
    print('sole_instance_ids: {}'.format(sole_instance_ids))

    for id in sole_instance_ids:
        print('instance {}: {}'.format(id, bdd.loadInstances(id)))

    # View annotations for specific video_id
    val_img_path = '/Users/pangjiangmiao/Documents/jiangmiao/images/val'
    view_save_path = '/Users/pangjiangmiao/Documents/jiangmiao/images/view'
    os.makedirs(view_save_path, exist_ok=True)

    instance_color_map = dict()
    confirm_tube = defaultdict(list)
    for index, img_id in enumerate(sole_img_ids):

        img_info = bdd.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        img = cv2.imread(os.path.join(val_img_path, file_name))

        ann_ids = bdd.getAnnIds(img_id)
        anns = bdd.loadAnns(ann_ids)

        for ann in anns:
            instance_id = ann['instance_id']
            if instance_id not in instance_color_map.keys():
                instance_color_map[instance_id] = randomcolor()

            if instance_id not in confirm_tube.keys():
                confirm_tube[instance_id] = defaultdict(list)
                confirm_tube[instance_id]['video_id'] = video_ids[1]
            confirm_tube[instance_id]['img_indexes'].append(index)
            confirm_tube[instance_id]['ann_ids'].append(ann['id'])

            color = instance_color_map[instance_id]
            x1 = int(ann['bbox'][0])
            y1 = int(ann['bbox'][1])
            x2 = int(ann['bbox'][0] + ann['bbox'][2] - 1)
            y2 = int(ann['bbox'][1] + ann['bbox'][3] - 1)

            if instance_id == 56743:
                cv2.rectangle(img, (x1, y1), (x2, y2), color)

        cv2.imwrite(os.path.join(view_save_path, file_name), img)

    print('Done')
    print(confirm_tube)


if __name__ == '__main__':
    main()

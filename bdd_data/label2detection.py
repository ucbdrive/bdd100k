import json
import os
import os.path as osp
import sys


def label2detection(label):
    boxes = list()
    for frame in label['frames']:
        for obj in frame['objects']:
            if 'box2d' not in obj:
                continue
            box = {'name': label['name'],
                   'timestamp': frame['timestamp'],
                   'category': obj['category'],
                   'bbox': obj['box2d']}
            boxes.append(box)
    return boxes


def change_dir(label_dir, detection_dir, phase='train'):
    label_dir = osp.join(label_dir, phase)
    if not osp.exists(label_dir):
        if phase in ['train', 'val']:
            print('Can not find', label_dir)
        return
    print('Processing', label_dir)
    input_names = [n for n in os.listdir(label_dir)
                   if osp.splitext(n)[1] == '.json']
    boxes = []
    count = 0
    for name in input_names:
        in_path = osp.join(label_dir, name)
        out = label2detection(json.load(open(in_path, 'r')))
        boxes.extend(out)
        count += 1
        if count % 1000 == 0:
            print('Finished', count)
    if not osp.exists(detection_dir):
        os.makedirs(detection_dir)
    out_path = osp.join(detection_dir, phase + '_bbox.json')
    with open(out_path, 'w') as fp:
        json.dump(boxes, fp, indent=4, separators=(',', ': '))


def main():
    phases = ['train', 'val', 'test']
    for p in phases:
        change_dir(sys.argv[1], sys.argv[2], p)


if __name__ == '__main__':
    main()

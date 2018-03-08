import json
import argparse
import os
import pdb
from os.path import exists, splitext, isdir, isfile, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
import sys


def parse_args():
    """Grab user supplied arguments using the argparse library."""

    # Use arparse to get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True,
                        help="input raw image", type=str)
    parser.add_argument("-l", "--label", required=True,
                        help="corresponding bounding box annotation "
                             "(json file)", type=str)
    parser.add_argument("-o", "--output_image", required=False,
                        help="output image file with bbox visualization",
                        type=str)
    args = parser.parse_args()

    # Check if the corresponding bounding box annotation exits
    is_valid_file(parser, args.image)
    is_valid_file(parser, args.label)
    assert (isdir(args.image) and isdir(args.label)) or \
           (isfile(args.image) and isfile(args.label)), \
        "input and label should be both folders or files"

    return args


def is_valid_file(parser, file_name):
    """Ensure that the file exists."""

    if not os.path.exists(file_name):
        parser.error("The corresponding bounding box annotation '{}' does "
                     "not exist!".format(file_name))
        sys.exit(1)


def get_areas(objects):
    return [o for o in objects
            if 'poly2d' in o and o['category'][:4] == 'area']


def get_lanes(objects):
    return [o for o in objects
            if 'poly2d' in o and o['category'][:4] == 'lane']


def get_boxes(objects):
    return [o for o in objects if 'box2d' in o]


def poly2patch(poly2d, closed=False, alpha=1):
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)

    # print(codes, points)
    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=random_color() if closed else 'none',
        edgecolor=random_color() if not closed else 'none',
        lw=0 if closed else 4, alpha=alpha)


def random_color():
    return np.random.rand(3)


def box2rect(box2d):
    """generate individual bounding box from label"""
    x1 = box2d['x1']
    y1 = box2d['y1']
    x2 = box2d['x2']
    y2 = box2d['y2']

    # Pick random color for each box
    box_color = random_color()
    
    # Draw and add one box to the figure
    return mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=3, edgecolor=box_color, facecolor='none', fill=False,
        alpha=0.75
    )


class LabelViewer(object):
    def __init__(self, args):
        """Visualize bounding boxes"""
        self.ax = None
        self.current_index = 0
        image_paths = [args.image]
        label_paths = [args.label]
        if isdir(args.label):
            input_names = [splitext(n)[0] for n in os.listdir(args.label) if
                           splitext(n)[1] == '.json']
            image_paths = [join(args.image, n + '.jpg') for n in input_names]
            label_paths = [join(args.label, n + '.json') for n in input_names]
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.font = FontProperties()
        self.font.set_family(['Luxi Mono', 'monospace'])

    def show(self):
        # Read and draw image
        dpi = 80
        w = 16
        h = 9
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        if len(self.image_paths) > 1:
            plt.connect('key_release_event', self.next_image)
        self.show_image()
        plt.show()

    def show_image(self):
        plt.cla()
        image_path = self.image_paths[self.current_index]
        label_path = self.label_paths[self.current_index]
        print('Image:', image_path)
        print('Label:', label_path)
        img = mpimg.imread(image_path)
        im = np.array(img, dtype=np.uint8)
        self.ax.imshow(im, interpolation='nearest', aspect='auto')

        # Read annotation labels
        with open(label_path) as data_file:
            label = json.load(data_file)
        objects = label['frames'][0]['objects']

        attributes = label['attributes']
        self.ax.text(
            25, 110,
            '  scene: {}\nweather: {}\n   time: {}'.format(
                attributes['scene'], attributes['weather'],
                attributes['timeofday']),
            fontproperties=self.font,
            fontsize=24,
            fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 10, 'lw': 0})

        [self.ax.add_patch(poly2patch(a['poly2d'], closed=True, alpha=0.5))
         for a in get_areas(objects)]
        [self.ax.add_patch(poly2patch(a['poly2d'], closed=False, alpha=0.75))
         for a in get_lanes(objects)]
        [self.ax.add_patch(box2rect(b['box2d'])) for b in get_boxes(objects)]
        self.ax.axis('off')

    def next_image(self, event):
        if event.key == 'n':
            self.current_index += 1
        elif event.key == 'p':
            self.current_index -= 1
        else:
            return
        self.current_index = max(min(self.current_index,
                                     len(self.image_paths) - 1), 0)
        self.show_image()
        plt.draw()


def main():
    args = parse_args()
    viewer = LabelViewer(args)
    viewer.show()


if __name__ == '__main__':
    main()

import json
import argparse
import pdb
from collections import namedtuple
from multiprocessing import Pool
import os
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from PIL import Image
import sys

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
]


def parse_args():
    """Grab user supplied arguments using the argparse library."""

    # Use argparse to get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True,
                        help="input raw image", type=str)
    parser.add_argument("-l", "--label", required=True,
                        help="corresponding bounding box annotation "
                             "(json file)", type=str)
    parser.add_argument('-s', '--scale', type=int, default=1,
                        help="Scale up factor for annotation factor. "
                             "Useful when producing visualization as "
                             "thumbnails.")
    parser.add_argument('--no-attr', action='store_true', default=False,
                        help="Do not show attributes")
    parser.add_argument('--no-lane', action='store_true', default=False,
                        help="Do not show lanes")
    parser.add_argument('--no-drivable', action='store_true', default=False,
                        help="Do not show drivable areas")
    parser.add_argument('--no-box2d', action='store_true', default=False,
                        help="Do not show 2D bounding boxes")
    parser.add_argument("-o", "--output_dir", required=False, default=None,
                        type=str,
                        help="output image file with bbox visualization. "
                             "If it is set, the images will be written to the "
                             "output folder instead of being displayed "
                             "interactively.")
    parser.add_argument('--instance', action='store_true', default=False,
                        help='Set instance segmentation mode')
    parser.add_argument('--drivable', action='store_true', default=False,
                        help='Set drivable area mode')
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

    if not exists(file_name):
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


def random_color():
    return np.random.rand(3)


def seg2color(seg):
    num_ids = 20
    train_colors = np.zeros((num_ids, 3), dtype=np.uint8)
    for l in labels:
        if l.trainId < 255:
            train_colors[l.trainId] = l.color
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(num_ids):
        color[seg == i, :] = train_colors[i]
    return color


def instance2color(instance):
    instance_colors = dict([(i, (np.random.random(3) * 255).astype(np.uint8))
                            for i in np.unique(instance)])
    color = np.zeros((instance.shape[0], instance.shape[1], 3), dtype=np.uint8)
    for k, v in instance_colors.items():
        color[instance == k] = v
    return color


def convert_instance_rgb(label_path):
    label_dir = dirname(label_path)
    label_name = splitext(split(label_path)[1])[0]
    image = np.array(Image.open(label_path, 'r'))
    seg = image[:, :, 0]
    seg_color = seg2color(seg)
    image = image.astype(np.uint32)
    instance = image[:, :, 0] * 1000 + image[:, :, 1]
    # instance_color = instance2color(instance)
    Image.fromarray(seg).save(
        join(label_dir, label_name + '_train_id.png'))
    Image.fromarray(seg_color).save(
        join(label_dir, label_name + '_train_color.png'))
    Image.fromarray(instance).save(
        join(label_dir, label_name + '_instance_id.png'))
    # Image.fromarray(instance_color).save(
    #     join(label_dir, label_name + '_instance_color.png'))


def drivable2color(seg):
    colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        color[seg == i, :] = colors[i]
    return color


def convert_drivable_rgb(label_path):
    label_dir = dirname(label_path)
    label_name = splitext(split(label_path)[1])[0]
    image = np.array(Image.open(label_path, 'r'))
    seg = image[:, :, 0]
    seg_color = drivable2color(seg)
    image = image.astype(np.uint32)
    instance = image[:, :, 0] * 1000 + image[:, :, 1]
    # instance_color = instance2color(instance)
    Image.fromarray(seg).save(
        join(label_dir, label_name + '_drivable_id.png'))
    Image.fromarray(seg_color).save(
        join(label_dir, label_name + '_drivable_color.png'))
    Image.fromarray(instance).save(
        join(label_dir, label_name + '_drivable_instance_id.png'))
    # Image.fromarray(instance_color).save(
    #     join(label_dir, label_name + '_drivable_instance_color.png'))


class LabelViewer(object):
    def __init__(self, args):
        """Visualize bounding boxes"""
        self.ax = None
        self.fig = None
        self.current_index = 0
        self.scale = args.scale
        image_paths = [args.image]
        label_paths = [args.label]
        if isdir(args.label):
            input_names = sorted(
                [splitext(n)[0] for n in os.listdir(args.label)
                 if splitext(n)[1] == '.json'])
            image_paths = [join(args.image, n + '.jpg') for n in input_names]
            label_paths = [join(args.label, n + '.json') for n in input_names]
        self.image_paths = image_paths
        self.label_paths = label_paths

        self.font = FontProperties()
        self.font.set_family(['Luxi Mono', 'monospace'])
        self.font.set_weight('bold')
        self.font.set_size(18 * self.scale)

        self.with_image = True
        self.with_attr = not args.no_attr
        self.with_lane = not args.no_lane
        self.with_drivable = not args.no_drivable
        self.with_box2d = not args.no_box2d
        self.with_segment = True

        self.out_dir = args.output_dir
        self.label_map = dict([(l.name, l) for l in labels])
        self.color_mode = 'random'

        self.image_width = 1280
        self.image_height = 720

        self.instance_mode = False
        self.drivable_mode = False
        self.with_post = False  # with post processing

        if args.drivable:
            self.set_drivable_mode()

        if args.instance:
            self.set_instance_mode()

    def view(self):
        self.current_index = 0
        if self.out_dir is None:
            self.show()
        else:
            self.write()

    def show(self):
        # Read and draw image
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        if len(self.image_paths) > 1:
            plt.connect('key_release_event', self.next_image)
        self.show_image()
        plt.show()

    def write(self):
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        out_paths = []
        for i in range(len(self.image_paths)):
            self.current_index = i
            self.show_image()
            out_name = splitext(split(self.image_paths[i])[1])[0] + '.png'
            out_path = join(self.out_dir, out_name)
            self.fig.savefig(out_path, dpi=dpi)
            out_paths.append(out_path)
        if self.with_post:
            print('Post-processing')
            p = Pool(10)
            if self.instance_mode:
                p.map(convert_instance_rgb, out_paths)
            if self.drivable_mode:
                p = Pool(10)
                p.map(convert_drivable_rgb, out_paths)

    def set_instance_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = False
        self.with_lane = False
        self.with_box2d = False
        self.with_segment = True
        self.color_mode = 'instance'
        self.instance_mode = True
        self.with_post = True

    def set_drivable_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = True
        self.with_lane = False
        self.with_box2d = False
        self.with_segment = False
        self.color_mode = 'instance'
        self.drivable_mode = True
        self.with_post = True

    def show_image(self):
        plt.cla()
        label_path = self.label_paths[self.current_index]
        name = splitext(split(label_path)[1])[0]
        print('Image:', name)
        self.fig.canvas.set_window_title(name)

        if self.with_image:
            image_path = self.image_paths[self.current_index]
            img = mpimg.imread(image_path)
            im = np.array(img, dtype=np.uint8)
            self.ax.imshow(im, interpolation='nearest', aspect='auto')
        else:
            self.ax.set_xlim(0, self.image_width - 1)
            self.ax.set_ylim(0, self.image_height - 1)
            self.ax.invert_yaxis()
            self.ax.add_patch(self.poly2patch(
                [[0, 0, 'L'], [0, self.image_height - 1, 'L'],
                 [self.image_width - 1, self.image_height - 1, 'L'],
                 [self.image_width - 1, 0, 'L']],
                closed=True, alpha=1., color='black'))

        # Read annotation labels
        with open(label_path) as data_file:
            label = json.load(data_file)
        objects = label['frames'][0]['objects']

        if 'attributes' in label and self.with_attr:
            attributes = label['attributes']
            self.ax.text(
                25 * self.scale, 90 * self.scale,
                '  scene: {}\nweather: {}\n   time: {}'.format(
                    attributes['scene'], attributes['weather'],
                    attributes['timeofday']),
                fontproperties=self.font,
                color='red',
                bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 10, 'lw': 0})

        if self.with_drivable:
            self.draw_drivable(objects)
        if self.with_lane:
            [self.ax.add_patch(self.poly2patch(
                a['poly2d'], closed=False, alpha=0.75))
             for a in get_lanes(objects)]
        if self.with_box2d:
            [self.ax.add_patch(self.box2rect(b['box2d']))
             for b in get_boxes(objects)]
        if self.with_segment:
            self.draw_segments(objects)
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

    def poly2patch(self, poly2d, closed=False, alpha=1., color=None):
        moves = {'L': Path.LINETO,
                 'C': Path.CURVE4}
        points = [p[:2] for p in poly2d]
        codes = [moves[p[2]] for p in poly2d]
        codes[0] = Path.MOVETO

        if closed:
            points.append(points[0])
            codes.append(Path.CLOSEPOLY)

        if color is None:
            color = random_color()

        # print(codes, points)
        return mpatches.PathPatch(
            Path(points, codes),
            facecolor=color if closed else 'none',
            edgecolor=color if not closed else 'none',
            lw=0 if closed else 2 * self.scale, alpha=alpha,
            antialiased=False)

    def draw_drivable(self, objects):
        objects = get_areas(objects)
        for obj in objects:
            if self.color_mode == 'random':
                color = random_color()
                alpha = 0.5
            else:
                color = (
                    (1 if obj['category'] == 'area/drivable' else 2) / 255.,
                    obj['id'] / 255., 0)
                alpha = 1
            self.ax.add_patch(self.poly2patch(
                obj['poly2d'], closed=True, alpha=alpha, color=color))

    def draw_segments(self, objects):
        color_mode = self.color_mode
        for obj in objects:
            if 'segments2d' not in obj:
                continue
            if color_mode == 'random':
                color = random_color()
                alpha = 0.5
            elif color_mode == 'instance':
                try:
                    label = self.label_map[obj['category']]
                    color = (label.trainId / 255., obj['id'] / 255., 0)
                except KeyError:
                    color = (1, 0, 0)
                alpha = 1
            else:
                raise ValueError('Unknown color mode {}'.format(
                    self.color_mode))
            for segment in obj['segments2d']:
                self.ax.add_patch(self.poly2patch(
                    segment, closed=True, alpha=alpha, color=color))

    def box2rect(self, box2d):
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
            linewidth=3 * self.scale, edgecolor=box_color, facecolor='none',
            fill=False, alpha=0.75
        )


def main():
    args = parse_args()
    viewer = LabelViewer(args)
    viewer.view()


if __name__ == '__main__':
    main()

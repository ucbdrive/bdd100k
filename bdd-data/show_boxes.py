import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import sys


def parse_args():
    """Grab user supplied arguments using the argparse library."""

    # Use arparse to get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", required=True,
                        help="input raw image", type=str)
    parser.add_argument("-b", "--box2d_labels", required=True,
                        help="corresponding bounding box annotation "
                             "(json file)", type=str)
    parser.add_argument("-o", "--output_image", required=True,
                        help="output image file with bbox visualization",
                        type=str)
    args = parser.parse_args()

    # Check if the corresponding bounding box annotation exits
    is_valid_file(parser, args.box2d_labels)

    return args.input_image, args.box2d_labels, args.output_image


def is_valid_file(parser, file_name):
    """Ensure that the file exists."""

    if not os.path.exists(file_name):
        parser.error("The corresponding bounding box annotation '{}' does "
                     "not exist!".format(file_name))
        sys.exit(1)


def visualize_bbox(args):
    """Visualize bounding boxes"""
    # Read and draw image
    dpi = 80
    w = 16
    h = 9
    img = mpimg.imread(args[0])
    im = np.array(img, dtype=np.uint8)

    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

    ax.imshow(im, interpolation='nearest', aspect='auto')

    # Read annotation labels
    data_path = args[1]
    with open(data_path) as data_file:
        labels = json.load(data_file)

    # Draw bounding boxes on image
    for label in labels['frames'][0]['objects']:
        ax.add_patch(generate_rect(label))

    ax.set_axis_off()

    # Visualize and save to output image
    fig.savefig(args[2], dpi=dpi)
    plt.show()


def generate_rect(label):
    """generate individual bounding box from label"""
    pos = label['box2d']
    x1 = pos['x1']
    y1 = pos['y1']
    x2 = pos['x2']
    y2 = pos['y2']

    # Pick random color for each box
    box_color = np.random.random((10, 3))[0]
    
    # Draw and add one box to the figure
    return patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor=box_color, facecolor='none', fill=False
    )


def main():
    args = parse_args()
    visualize_bbox(args)


if __name__ == '__main__':
    main()

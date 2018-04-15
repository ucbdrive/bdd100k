# BDD Data

This is supporting code for [BDD Data](http://bdd-data.berkeley.edu/). Please check the data download on the homepage to obtain the dataset. This code supports BDD100K, in particular.

## Understanding the Data

After being unzipped, all the files will reside in a folder named `bdd100k`. All the original videos are in `bdd100k/videos` and labels in `bdd100k/labels`. `bdd100k/images` contains the frame at 10th second in the corresponding video.

`bdd100k/labels` contains json files based on [our label format](doc/format.md). [`bdd_data/show_labels.py`](bdd_data/show_labels.py) provides examples to parse and visualize the labels.

For example, you can view training data one by one

```
python3 bdd_data/show_labels.py -i bdd100k/images/100k/train -l bdd100k/labels/100k/train
```

Or export the drivable area in segmentation maps:

```
python3 bdd_data/show_labels.py -i bdd100k/images/100k/train -l bdd100k/labels/100k/train \
    -s 1 -o bdd100k/out_drivable_maps/train --drivable
```

This exporting process will take a while, so we also provide `Drivable Maps` in the downloading page, which will be `bdd100k/drivable_maps` after decompressing.

### Object Detection

You can export object detection in concise format by

```
python3 bdd_data/label2detection.py bdd100k/labels/100k bdd100k/detection
```

The detection label format is below, which is the same as [our detection evaluation format](doc/evaluation.md#):

```
[
   {
      "name": str,
      "timestamp": 1000,
      "category": str,
      "bbox": [x1, y1, x2, y2],
      "score": float
   }
]
```

### Semantic Segmentation

At present time, instance segmentation is provided as semantic segmentation maps and polygons in json will be provided in the future. The encoding of labels should still be `train_id` defined in [`bdd_data/label.py`](bdd_data/label.py), thus car should be 13.
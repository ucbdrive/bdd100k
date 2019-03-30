# BDD Data

This is supporting code for [BDD100K data](http://bdd-data.berkeley.edu/) and [Scalabel](https://www.scalabel.ai/).

Please check the data download on the homepage to obtain the dataset. This code supports BDD100K, in particular.


![teaser](doc/teaser.png)

## Dependency

- Python 3
- `pip3 install -r requirements.txt`

## Understanding the Data

After being unzipped, all the files will reside in a folder named `bdd100k`. All the original videos are in `bdd100k/videos` and labels in `bdd100k/labels`. `bdd100k/images` contains the frame at 10th second in the corresponding video.

`bdd100k/labels` contains two json files based on [our label format](doc/format.md) for training and validation sets. [`bdd_data/show_labels.py`](bdd_data/show_labels.py) provides examples to parse and visualize the labels.

For example, you can view training data one by one

```
python3 -m bdd_data.show_labels.py --image-dir bdd100k/images/100k/train \
    -l bdd100k/labels/bdd100k_labels_images_train.json
```

Or export the drivable area in segmentation maps:

```
python3 -m bdd_data.show_labels.py --image-dir bdd100k/images/100k/train \
    -l bdd100k/labels/bdd100k_labels_images_train.json \
    -s 1 -o bdd100k/out_drivable_maps/train --drivable
```

This exporting process will take a while, so we also provide `Drivable Maps` in the downloading page, which will be `bdd100k/drivable_maps` after decompressing. There are 3 possible labels on the maps: 0 for background, 1 for direct drivable area and 2 for alternative drivable area.

### Trajectories

To visualize the GPS trajectories provided in `bdd100k/info`, you can run the command below to produce an html file that displays a single trajectory and output the results in folder `out/`:

```
python3 -m bdd_data.show_gps_trajectory.py \
    -i bdd100k/info/train/0000f77c-6257be58.json -o out/ -k {YOUR_API_KEY}
```

Or create html file for each GPS trajectory in a directory, for example:

```
python3 -m bdd_data.show_gps_trajectory.py \
    -i bdd100k/info/train -o out/ -k {YOUR_API_KEY}
```

To create a Google Map API key, please follow the instruction [here](https://developers.google.com/maps/documentation/embed/get-api-key). The generated maps will look like

![gps_trajectory](doc/trajectory_gmap.jpg)

### Object Detection

You can export object detection in concise format by

```
python3 -m bdd_data.label2det.py bdd100k/labels/bdd100k_labels_images_train.json \
    bdd100k/detection_train.json
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

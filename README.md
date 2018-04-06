View training data one by one

```
python3 show_labels.py -i images/ -l labels_100k_release/train/
```

Generate drivable area segmentation
```
python3 show_labels.ph -i images/ -l labels_100k_release/train/ -s 1 -o drivable_100K/train --drivable
```

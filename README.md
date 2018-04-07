View training data one by one

```
python3 show_labels.py -i bdd100k/images/100k/train -l bdd100k/labels/100k/train
```

Generate drivable area segmentation
```
python3 show_labels.ph -i bdd100k/images/100k/train -l bdd100k/labels/100k/train -s 1 -o bdd100k/out_drivable_maps/train --drivable
```

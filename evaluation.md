## Road Object Detection

We use the following 3 metrics to evaluate the performance of detection:

#### Average Precision (AP):

| Metric | |
|:---:|---|
| AP | % AP at IoU=.50:.05:.95 (primary challenge metric) |
| APIoU=.50 | % AP at IoU=.50 (PASCAL VOC metric) |
| APIoU=.75 | % AP at IoU=.75 (strict metric) |


#### Submission format

The entire result struct array is stored as a single JSON file (save via gason in Matlab or json.dump in Python).

```
[{
"name" : str, "timestamp": 1000, "category" : str, "bbox" : [x1,y1,x2,y2], "score" : float,
}]
```

Box coordinates are integers measured from the top left image corner (and are 0-indexed). `[x1, y1]` is the top left corner of the bounding box and `[x2, y2]` the lower right. `name` is the video name that the frame is extracted from. It composes of two 8-character identifiers connected '-', such as `c993615f-350c682c`. Candidates for `category` are `['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider']`. In the current data, all the image timestamps are 1000.


## Segmentation

Both drivable area and semantic segmentation follow the same evaluation metric.

Following the practice of [Cityscapes](http://www.cityscapes-dataset.com) challenge, we calculate the intersection-over-union metric from [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) across the whole test set, IoU=true positive/true positive+false positive+false negative.

Result files with filename "XXX*.png" where XXX is the corresponding name of test video (19-character identifier). The image size of results should be equal to the input image size. The encoding of labels should still be `train_id`, thus car should be 13.




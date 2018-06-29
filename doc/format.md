## 2D Bounding Box

- name: string
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- frames [ ]:
    - timestamp: int64 (epoch time ms)
    - index: int (optional, frame index in this video)
    - objects [ ]:
        - id: int32
        - category: string (classification)
        - attributes:
            - occluded: boolean
            - truncated: boolean
            - trafficLightColor: "red|green|yellow|none"
            - direction: "parallel|vertical" (for lanes)
            - style: "solid | dashed" (for lanes)
        - box2d:
            - x1: pixels
            - y1: pixels
            - x2: pixels
            - y2: pixels
        - poly2d: Each segment is an array of 2D points with type (array)
                  "L" means line and "C" means beizer curve.
        - seg2d: List of poly2d. Some object segmentation may contain multiple regions




Road object categories:
```
[
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]
```
They are labeld by `box2d`.

Drivable area categories:
```
[
    "area/alternative",
    "area/drivable"
]
```

Lane marking categories:
```
[
    "lane/crosswalk",
    "lane/double other",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow"
]
```

Both drivable areas and lane markings are labeled by `poly2d`. Please check the visulization code [`show_labels.py`](../bdd_data/show_labels.py) for examples of drawing all the labels.

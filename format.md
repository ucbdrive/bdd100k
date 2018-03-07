## 2D Bounding Box

- name: string
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
        - box2d:
            - x1: pixels
            - y1: pixels
            - x2: pixels
            - y2: pixels
        - poly2d: an array of 2D points with type (array)
                  "L" means line and "C" means beizer curve
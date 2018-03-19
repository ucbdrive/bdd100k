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
        - segments: List of segments. Each segment is an array of 2D points with type (array)
                  "L" means line and "C" means beizer curve. Some object segmentation may contain multiple regions
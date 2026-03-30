from dataclasses import dataclass

@dataclass
class TrackPoint:
    frame: int
    x: int
    y: int
    radius: float
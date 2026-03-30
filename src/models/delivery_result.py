from dataclasses import dataclass, field

@dataclass
class DeliveryResult:
    ball_no: int
    bounced: bool
    length: str
    bounce_frame: int
    bounce_point: tuple
    start_frame: int
    end_frame: int
    duration_s: float
    tracked_points: int
    track: list = field(default_factory=list)
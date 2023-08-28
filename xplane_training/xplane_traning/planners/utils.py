from dataclasses import dataclass

@dataclass
class MPCCost:
    total: float
    tracking: float
    control: float
    goal: float

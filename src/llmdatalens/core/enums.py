from enum import Enum, auto

class MetricField(str, Enum):
    Accuracy = "accuracy"
    Performance = "performance"
    Confidence = "confidence"
    Robustness = "robustness"
    Consistency = "consistency"
    Other = "other"

    @classmethod
    def _missing_(cls, value):
        return cls.Other

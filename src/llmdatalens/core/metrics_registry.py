from typing import Dict, Callable, Any, Optional, List
from functools import wraps
import inspect
import re
from llmdatalens.core.enums import MetricField

def is_pascal_case(s: str) -> bool:
    return re.match(r'^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*$', s) is not None

class MetricInfo:
    def __init__(self, func: Callable, description: str, field: MetricField, input_keys: List[str]):
        self.func = func
        self.description = description
        self.field = field
        self.input_keys = input_keys

class MetricsRegistry:
    _instance = None
    _registry: Dict[str, MetricInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, field: MetricField = MetricField.Other, input_keys: List[str] = []):
        if not is_pascal_case(name):
            raise ValueError(f"Metric name '{name}' is not in PascalCase.")
        
        def decorator(func: Callable):
            description = inspect.getdoc(func) or "No description provided"
            cls._registry[name] = MetricInfo(func, description.strip(), field, input_keys or [])
            setattr(MetricNames, name, name)  # Add the metric name to MetricNames class
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[MetricInfo]:
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, MetricInfo]:
        return cls._registry

class MetricNames:
    pass

metrics_registry = MetricsRegistry()

def register_metric(name: str, field: MetricField = MetricField.Other, input_keys: List[str] = []):
    return metrics_registry.register(name, field, input_keys)

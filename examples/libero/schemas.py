import csv
import json
import pathlib
from dataclasses import dataclass, field, fields, asdict
from typing import List, Type, TypeVar

T = TypeVar("T", bound="CSVDataclass")
J = TypeVar("J", bound="JSONDataclass")


class CSVDataclass:
    """Mixin class that adds CSV serialization to dataclasses."""

    @classmethod
    def to_csv(cls: Type[T], instances: List[T], filepath: pathlib.Path) -> None:
        """Save a list of dataclass instances to a CSV file."""
        if not instances:
            return

        with open(filepath, "w", newline="") as f:
            fieldnames = [field.name for field in fields(cls)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for instance in instances:
                writer.writerow(
                    {field.name: getattr(instance, field.name) for field in fields(cls)}
                )

    @classmethod
    def from_csv(cls: Type[T], filepath: pathlib.Path) -> List[T]:
        """Load a list of dataclass instances from a CSV file."""
        instances = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types based on field annotations
                kwargs = {}
                for field in fields(cls):
                    value = row[field.name]
                    # Handle type conversion
                    if field.type in (int, "int"):
                        kwargs[field.name] = int(value)
                    elif field.type in (float, "float"):
                        kwargs[field.name] = float(value)
                    elif field.type in (bool, "bool"):
                        kwargs[field.name] = value.lower() in ("true", "1", "yes")
                    else:
                        kwargs[field.name] = value
                instances.append(cls(**kwargs))
        return instances


class JSONDataclass:
    """Mixin class that adds JSON serialization to dataclasses."""

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        """Save a dataclass instance to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=indent)

    @classmethod
    def from_json(cls: Type[J], filepath: pathlib.Path) -> J:
        """Load a dataclass instance from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            return cls(**data)


@dataclass
class ActionChunk(CSVDataclass):
    chunk_length: int
    request_timestamp: float
    response_timestamp: float
    start_step: int = field(default_factory=lambda: -1)

    def set_start_step(self, start_step: int) -> None:
        self.start_step = start_step


@dataclass
class Timestamp(CSVDataclass):
    timestamp: float
    action_chunk_index: int
    action_chunk_current_step: int

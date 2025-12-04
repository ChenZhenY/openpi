"""Base class for dataclasses with CSV serialization."""

import csv
import pathlib
from dataclasses import fields
from typing import List, TypeVar

T = TypeVar("T", bound="CSVDataclass")


class CSVDataclass:
    """Mixin class that adds CSV serialization to dataclasses."""

    @classmethod
    def to_csv(cls: type[T], instances: List[T], filepath: pathlib.Path) -> None:
        """Save a list of dataclass instances to a CSV file."""
        if not instances:
            return

        with open(filepath, "w", newline="") as f:
            fieldnames = [field.name for field in fields(cls)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for instance in instances:
                writer.writerow({field.name: getattr(instance, field.name) for field in fields(cls)})

    @classmethod
    def from_csv(cls: type[T], filepath: pathlib.Path) -> List[T]:
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

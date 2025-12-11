import csv
import json
import pathlib
from dataclasses import dataclass, field, fields, asdict
from typing import List, Type, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T", bound="CSVDataclass")
J = TypeVar("J", bound="JSONDataclass")
P = TypeVar("P", bound="ParquetDataclass")


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


class ParquetDataclass:
    """Mixin class that adds Parquet serialization to dataclasses."""

    @classmethod
    def to_parquet(cls: Type[P], instances: List[P], filepath: pathlib.Path) -> None:
        """Save a list of dataclass instances to a Parquet file."""
        if not instances:
            return

        # Convert instances to dictionary format
        data_dict = {field.name: [] for field in fields(cls)}

        for instance in instances:
            for f in fields(cls):
                value = getattr(instance, f.name)
                data_dict[f.name].append(value)

        # Convert lists of numpy arrays to a format Parquet can handle
        for f in fields(cls):
            values = data_dict[f.name]
            if values and isinstance(values[0], np.ndarray):
                # Convert to list of lists for nested array storage
                data_dict[f.name] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in values
                ]

        # Create DataFrame and write to Parquet
        df = pd.DataFrame(data_dict)
        df.to_parquet(filepath, engine="pyarrow", index=False)

    @classmethod
    def from_parquet(cls: Type[P], filepath: pathlib.Path) -> List[P]:
        """Load a list of dataclass instances from a Parquet file."""
        df = pd.read_parquet(filepath, engine="pyarrow")

        instances = []
        for _, row in df.iterrows():
            kwargs = {}
            for f in fields(cls):
                value = row[f.name]

                # Convert back to numpy arrays if needed
                # This is a heuristic - you might want to add field annotations
                # to specify which fields should be numpy arrays
                if isinstance(value, list) and value and isinstance(value[0], list):
                    kwargs[f.name] = np.array(value)
                elif pd.isna(value):
                    kwargs[f.name] = None
                else:
                    kwargs[f.name] = value

            instances.append(cls(**kwargs))

        return instances


@dataclass
class ActionChunk(ParquetDataclass):
    actions: List[List[float]]
    request_timestamp: float
    response_timestamp: float
    start_step: int = field(default_factory=lambda: -1)

    def set_start_step(self, start_step: int) -> None:
        self.start_step = start_step

    @property
    def latency(self) -> float:
        return self.response_timestamp - self.request_timestamp

    @property
    def chunk_length(self) -> int:
        return len(self.actions)

    def get_action(self, index: int) -> List[float]:
        return self.actions[index]


@dataclass
class Timestamp(CSVDataclass):
    timestamp: float
    action_chunk_index: int
    action_chunk_current_step: int

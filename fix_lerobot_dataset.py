#!/usr/bin/env python3
"""
Script to inspect and fix parquet metadata that contains 'List' feature types
"""
import json
import pyarrow.parquet as pq
from pathlib import Path
import sys

def inspect_parquet_metadata(parquet_path):
    """Inspect the metadata in parquet files"""
    parquet_dir = Path(parquet_path)
    
    if not parquet_dir.exists():
        print(f"Error: Path {parquet_path} does not exist")
        return
    
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {parquet_path}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Check the first parquet file
    first_file = parquet_files[0]
    print(f"\nInspecting: {first_file.name}")
    
    parquet_file = pq.ParquetFile(first_file)
    schema = parquet_file.schema_arrow
    
    print("\n=== Arrow Schema ===")
    print(schema)
    
    if schema.metadata:
        print("\n=== Schema Metadata Keys ===")
        for key in schema.metadata.keys():
            print(f"  {key}")
        
        if b'huggingface' in schema.metadata:
            print("\n=== HuggingFace Metadata ===")
            hf_metadata = json.loads(schema.metadata[b'huggingface'])
            print(json.dumps(hf_metadata, indent=2))
            
            # Check for 'List' in features
            if 'info' in hf_metadata and 'features' in hf_metadata['info']:
                features = hf_metadata['info']['features']
                print("\n=== Checking for 'List' type ===")
                find_list_types(features)

def find_list_types(obj, path=""):
    """Recursively find 'List' type in nested dictionary"""
    if isinstance(obj, dict):
        if '_type' in obj and obj['_type'] == 'List':
            print(f"Found 'List' type at: {path}")
            print(f"  Full object: {obj}")
        
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            find_list_types(value, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            find_list_types(item, new_path)

def fix_parquet_metadata(parquet_path, output_path=None):
    """Fix parquet metadata by replacing 'List' with 'Sequence'"""
    parquet_dir = Path(parquet_path)
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = parquet_dir
    
    for parquet_file in parquet_files:
        print(f"\nProcessing: {parquet_file.name}")
        
        # Read the parquet file
        table = pq.read_table(parquet_file)
        schema = table.schema
        
        # Get metadata
        metadata = dict(schema.metadata) if schema.metadata else {}
        
        if b'huggingface' in metadata:
            hf_metadata = json.loads(metadata[b'huggingface'])
            
            # Replace 'List' with 'Sequence'
            modified = replace_list_with_sequence(hf_metadata)
            
            if modified:
                print("  Fixed 'List' types")
                metadata[b'huggingface'] = json.dumps(hf_metadata).encode()
                
                # Create new schema with updated metadata
                new_schema = schema.with_metadata(metadata)
                new_table = table.cast(new_schema)
                
                # Write to output
                output_file = output_dir / parquet_file.name
                pq.write_table(new_table, output_file)
                print(f"  Wrote fixed file to: {output_file}")
            else:
                print("  No 'List' types found, skipping")
        else:
            print("  No HuggingFace metadata found")

def replace_list_with_sequence(obj):
    """Recursively replace 'List' with 'Sequence' in dictionary"""
    modified = False
    
    if isinstance(obj, dict):
        if '_type' in obj and obj['_type'] == 'List':
            obj['_type'] = 'Sequence'
            modified = True
        
        for value in obj.values():
            if replace_list_with_sequence(value):
                modified = True
    
    elif isinstance(obj, list):
        for item in obj:
            if replace_list_with_sequence(item):
                modified = True
    
    return modified

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspect: python fix_parquet_metadata.py <parquet_dir>")
        print("  Fix:     python fix_parquet_metadata.py <parquet_dir> --fix [output_dir]")
        sys.exit(1)
    
    parquet_path = sys.argv[1]
    
    if "--fix" in sys.argv:
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        fix_parquet_metadata(parquet_path, output_path)
    else:
        inspect_parquet_metadata(parquet_path)
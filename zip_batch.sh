#!/bin/bash

# Usage: ./zip_batches.sh /path/to/pt/files

INPUT_DIR="$1"
OUTPUT_DIR="abc"
BATCH_SIZE=100

# Check if input path is valid
if [ ! -d "$INPUT_DIR" ]; then
  echo "Directory $INPUT_DIR does not exist."
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Python to generate batches
python3 - <<EOF
import os
import zipfile

input_dir = "$INPUT_DIR"
output_dir = "$OUTPUT_DIR"
batch_size = $BATCH_SIZE

# Get all .pt files
pt_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
pt_files.sort()

for i in range(0, len(pt_files), batch_size):
    batch = pt_files[i:i+batch_size]
    zip_name = os.path.join(output_dir, f"batch_{i//batch_size + 1:02}.zip")
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for f in batch:
            zipf.write(os.path.join(input_dir, f), arcname=f)
    print(f"Created: {zip_name} with {len(batch)} files.")
EOF
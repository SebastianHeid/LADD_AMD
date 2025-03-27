#!/bin/bash

# Directories to compare
dir1="/export/data/vislearn/rother_subgroup/sheid/LAION_LADD/txt_embs"
dir2="/export/data/vislearn/rother_subgroup/sheid/LAION_LADD/noises"
output_file="differences.txt"

# Ensure both directories exist
if [ ! -d "$dir1" ] || [ ! -d "$dir2" ]; then
    echo "Both arguments must be valid directories."
    exit 1
fi

# List elements in both directories
files_dir1=$(ls "$dir1")
files_dir2=$(ls "$dir2")

# Compare elements
echo "Files in $dir1 but not in $dir2:" > "$output_file"
comm -23 <(echo "$files_dir1" | sort) <(echo "$files_dir2" | sort) >> "$output_file"

echo "Files in $dir2 but not in $dir1:" >> "$output_file"
comm -13 <(echo "$files_dir1" | sort) <(echo "$files_dir2" | sort) >> "$output_file"

echo "Comparison completed. Results saved in $output_file."
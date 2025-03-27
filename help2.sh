#!/bin/bash

# Directory to delete files from
dir="/export/data/vislearn/rother_subgroup/sheid/LAION_LADD/latents"
diff_file="differences.txt"

# Ensure the directory exists
if [ ! -d "$dir" ]; then
    echo "The provided argument must be a valid directory."
    exit 1
fi

# Ensure the differences file exists
if [ ! -f "$diff_file" ]; then
    echo "The file $diff_file does not exist."
    exit 1
fi

# Read the differences file and delete the listed files
while IFS= read -r file; do
    file_path="$dir/$file"
    if [ -e "$file_path" ]; then
        rm -rf "$file_path"
        echo "Deleted: $file_path"
    else
        echo "Not found: $file_path"
    fi
done < "$diff_file"

echo "Deletion process completed."
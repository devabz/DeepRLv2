#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Directory to search for MP4 files (provided as the first argument)
rootDir="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$rootDir" ]; then
    echo "Error: Directory $rootDir does not exist."
    exit 1
fi

# Find all MP4 files and process them
find "$rootDir" -type f -name "*.mp4" | while read -r file; do
    # Get the directory of the MP4 file
    dir=$(dirname "$file")

    # Get the filename without the extension
    filename=$(basename "$file" .mp4)

    # Set the output GIF file path
    outputFile="$dir/$filename.gif"

    # Convert MP4 to GIF using ffmpeg
    ffmpeg -i "$file" -vf "fps=10,scale=320:-1:flags=lanczos" -c:v gif "$outputFile" -n
done

echo "Conversion complete."

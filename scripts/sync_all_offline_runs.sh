#!/bin/bash

# the script is generated by ChatGPT

# Set the directory path and file name
dir="wandb"
filename="offline_directories.txt"

# Use the find command to search for directories
# whose name starts with "offline", and save the results to a file
find "$dir" -type d -name "offline*" -print > "$filename"

# Loop through the file and call wandb sync on each directory
while read -r line; do
    wandb sync "$line"
done < "$filename"

# Remove the file when done
rm "$filename"

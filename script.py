import os
import shutil

def collect_unique_files_by_name(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    seen_names = set()  # Store seen file names to avoid duplicates

    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename not in seen_names:
                seen_names.add(filename)
                # Copy file to the destination folder
                shutil.copy(os.path.join(root, filename), os.path.join(destination_folder, filename))

source_directory = 'D:\\Semester 8\\Taylor Swift'  # Adjust to your source folder path
destination_directory = 'D:\\Semester 8\\TS_unique_2'  # Adjust to your destination folder path

collect_unique_files_by_name(source_directory, destination_directory)

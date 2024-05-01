import csv
import re

def natural_sort_key(s):
    """A key to sort strings that contain numbers by their embedded numeric value."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_and_sort_identifiers_from_file(filename):
    # Initialize a set to store unique identifiers
    unique_identifiers = set()

    # Open the file and use csv.reader to parse it
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        for row in reader:
            sub1, sub2 = row[0], row[1]
            unique_identifiers.update([sub1, sub2])  # Keep original case for now

    # Sort the identifiers using natural sort
    sorted_identifiers = sorted(unique_identifiers, key=natural_sort_key)
    return sorted_identifiers

# Specify the filename that contains your data

filename = 'C:\\Users\\mayal\\PlagiarismDetector\\dataset\\labels.csv'

# Call the function with the filename and print the results
identifiers = extract_and_sort_identifiers_from_file(filename)
print(identifiers)

def detect_repeated_tuples(file_path):
    seen_tuples = set()
    repeated_tuples = set()

    with open(file_path, 'r') as file:
        for line in file:
            # Clean and split the line into parts
            parts = line.strip().split()
            if len(parts) == 3:  # Ensure the line is correctly formatted
                tuple_key = tuple(parts)  # Create a tuple from the line
                if tuple_key in seen_tuples:
                    repeated_tuples.add(tuple_key)  # Add to repeated if already seen
                else:
                    seen_tuples.add(tuple_key)  # Add to seen if not already there

    return repeated_tuples

# Specify the path to your data file
file_path = 'C:\\Users\\mayal\\PlagiarismDetector\\dataset\\SOCO14-java.qrel'
repeated = detect_repeated_tuples(file_path)

# Output the repeated tuples
print("Repeated tuples:")
for tup in repeated:
    print(tup)

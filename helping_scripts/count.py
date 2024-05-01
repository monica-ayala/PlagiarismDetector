import csv

def count_verdicts(file_path):
    count_0 = 0
    count_1 = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['verdict'] == '0':
                count_0 += 1
            elif row['verdict'] == '1':
                count_1 += 1
    return count_0, count_1

# Path to your CSV file
file_path = 'C:\\Users\\mayal\\PlagiarismDetector\\dataset\\labels.csv'
count_0, count_1 = count_verdicts(file_path)
print(f"Occurrences of 0: {count_0}")
print(f"Occurrences of 1: {count_1}")

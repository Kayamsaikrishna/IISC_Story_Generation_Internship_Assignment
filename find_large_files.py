import os

# Path to your repository
repo_path = r"D:\IISC_INTERNSHIP\AI_Story_Generator"

# List to hold (file_path, size_in_MB)
large_files = []

# Walk through all files
for root, dirs, files in os.walk(repo_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if size_mb > 50:  # Only list files > 50 MB (change if needed)
                large_files.append((file_path, size_mb))
        except FileNotFoundError:
            pass  # Skip if file is inaccessible

# Sort by size (largest first)
large_files.sort(key=lambda x: x[1], reverse=True)

# Print results
print("\nLargest files in repo:")
for file_path, size_mb in large_files:
    print(f"{size_mb:.2f} MB  -  {file_path}")

import os

# Define folder structure
folders = [
    "src",
    "research",
    "data"
]

# Define files to be created
files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "research/trials.ipynb",
    "app.py",
    "setup.py",
    ".env",
    "requirements.txt"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create files
for file_path in files:
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass
        print(f"Created file: {file_path}")
    else:
        print(f"Already exists: {file_path}")

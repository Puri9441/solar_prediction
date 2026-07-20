import hashlib
import os

def generate_dataset_hash(file_path):
    """Generates an immutable hash for the dataset file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

if __name__ == '__main__':
    csv_path = os.path.join('data', 'train.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset file not found: {csv_path}')

    dataset_hash = generate_dataset_hash(csv_path)
    print(f"Dataset Lineage Hash: {dataset_hash}")

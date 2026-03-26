import sys
import os

# Ensure src is in path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoaderFactory

if __name__ == "__main__":
    print("Initiating SST-2 Data Download / Cache Verification...")
    factory = DataLoaderFactory(dataset_name="sst2")
    print(f"Number of training samples: {len(factory.dataset['train'])}")
    print(f"Number of validation samples: {len(factory.dataset['validation'])}")
    print("Data download step complete!")

import sys, os
import zipfile
import argparse
from pathlib import Path

parser = argparse.ArgumentParser ('ADC Dataset utils')
parser.add_argument('-e', default=False, action='store_true')

args = parser.parse_args()

# Finds exact path of dataset source in reference to this file's path
DATASET_SOURCE_PATH = Path(__file__).parent / '../..' / 'dataset' / 'ADC_Dataset.zip'

# Finds exact path of data source in reference to this file's path and creates one if it doesn't exist
DATA_DEST_PATH = Path(__file__).parent / '../..' / 'data'
if not Path.exists(DATA_DEST_PATH):
    os.mkdir(DATA_DEST_PATH)
    print("Creating data folder")

def extract_adc_dataset ():
    
    with zipfile.ZipFile (DATASET_SOURCE_PATH, 'r') as dataset_zip:
        dataset_zip.extractall(DATA_DEST_PATH)


if __name__ == "__main__":
    
    if args.e:    
        extract_adc_dataset()
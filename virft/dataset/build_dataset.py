import time
from datasets import DatasetDict, Dataset
from PIL import Image
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ORIC JSON to HF dataset")

    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to ORIC-style Train JSON file.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory of images.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save processed dataset.")

    return parser.parse_args()


def json_to_dataset(json_file_path, image_root):
    # read json file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    image_paths = [os.path.join(image_root, item['image']) for item in data]
    
    problems = [item['problem'] for item in data]
    solutions = [item['solution'] for item in data]

    images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]

    dataset_dict = {
        'image': [[p] for p in image_paths],   
        'problem': problems,                   
        'solution': solutions
    }


    dataset = Dataset.from_dict(dataset_dict)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict



def save_dataset(dataset_dict, save_path):
    # save DatasetDict to your disk
    dataset_dict.save_to_disk(save_path)


def load_dataset(save_path):
    # load DatasetDict
    return DatasetDict.load_from_disk(save_path)


def main(args):
    print("Start time:", time.asctime())

    # Load JSON and build dataset
    dataset_dict = json_to_dataset(args.json_path, args.image_dir)

    # Save to HF dataset format
    save_dataset(dataset_dict, args.save_path)

    print("Dataset saved. Checking load...")
    test_dataset = load_dataset(args.save_path)
    print("Loaded dataset:", test_dataset)

    print("Finish time:", time.asctime())


if __name__ == "__main__":
    args = parse_args()
    main(args)

import os
import json
import random
import argparse

from pycocotools.coco import COCO
from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np

from utils.oric import ORIC, DecodingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./dataset",
        help="Root folder containing COCO data; expects 'instances_val2014.json' under data_folder/coco.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./dataset",
        help="Directory to save cached files and final Q&A.",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=2,
        help="Number of objects per image for positive/negative questions.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of image pairs to sample (will generate ~2*num_images*num_objects Q&A).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-5-2025-08-07",
        help="Model to do LLM-guided sampling",
    )
    parser.add_argument(
        "--reject_prompt",
        type=str,
        default="./prompts/reject_sample.txt",
        help="Path to the reject_sample prompt template.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to use: train or val.",
    )
    return parser.parse_args()


def main(args):
    # Prepare output folder
    split = args.split
    os.makedirs(args.output_folder, exist_ok=True)
    # Build file paths
    sampled_images_path = os.path.join(args.output_folder, "sampled_images.json")
    sim_pairs_path = os.path.join(args.output_folder, "similar_pairs.json")
    sample_sim_path = os.path.join(args.output_folder, "sampled_similar_images_pairs.json")
    embeddings_path = os.path.join(args.output_folder, "embeddings.pt")
    if "val" == split:
        questions_path = os.path.join(args.output_folder, "oric_bench.json")
    else:
        questions_path = os.path.join(args.output_folder, "oric_train.json")

    # Initialize COCO and CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = COCO(os.path.join(args.data_folder, f"instances_{split}2014.json"))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Initialize GPT arguments
    decoding_args = DecodingArguments(
        model=args.llm_model, max_tokens=1024, image_detail="auto", temperature=0, top_p=1.0
    )

    # Initialize ORIC helper
    oric = ORIC(
        coco,
        model,
        processor,
        device,
        image_folder=os.path.join(args.data_folder, f"{split}2014"),
        reject_prompt_template=args.reject_prompt,
        split=split,
        decoding_args=decoding_args
    )

    # Sample images (cache to JSON)
    if os.path.exists(sampled_images_path):
        with open(sampled_images_path) as f:
            sampled_images = json.load(f)
    else:
        sampled_images = oric.extract_images(min_num_objects=3)
        with open(sampled_images_path, "w") as f:
            json.dump(sampled_images, f, indent=4)

    if os.path.exists(sample_sim_path):
        with open(sample_sim_path) as f:
            sim_pairs = json.load(f)
    else:
        # Find similar pairs (this method itself caches to JSON)
        sim_pairs = oric.extract_similar_images(
            sampled=sampled_images,
            embedding_path=embeddings_path,
            output_path=sim_pairs_path,
            batch_size=256,
        )

    # Reproducible sampling of pairs
    random.seed(args.seed)
    np.random.seed(42)
    if len(sim_pairs) > args.num_images:
        sim_pairs = random.sample(sim_pairs, args.num_images)

    # Build Q&A
    questions = oric.extract_QA(
        sim_pairs, num_targets=args.num_objects, max_images=args.num_images
    )
    
    # Save final questions
    with open(questions_path, "w") as f:
        json.dump(questions, f, indent=4)

    print(f"Done! Generated {len(questions)} Q&A entries → {questions_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

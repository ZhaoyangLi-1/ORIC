# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is adapted from Visual-RFT (https://github.com/Liuziyu77/Visual-RFT)
# with modifications for ORIC training, image-conditioned prompts, and custom reward functions

import os
import sys
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from math_verify import parse, verify
from virft.open_r1.trainer import Qwen3VLGRPOTrainer, Qwen3VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def _join_completion(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for x in completion:
            if isinstance(x, dict) and "content" in x and isinstance(x["content"], str):
                parts.append(x["content"])
            elif isinstance(x, str):
                parts.append(x)
        return "".join(parts)
    return str(completion)


def format_reward(completions, **kwargs):
    pattern = r"^\s*<REASONING>[\s\S]*?</REASONING>\s*<SOLUTION>[\s\S]*?</SOLUTION>\s*$"
    scores = []
    for completion in completions:
        content = _join_completion(completion)

        ok = bool(re.fullmatch(pattern, content))
        score = 1.0 if ok else 0.0

        if ok:
            if len(re.findall(r"</SOLUTION>", content)) != 1 or len(re.findall(r"</REASONING>", content)) != 1:
                score = 0.0

        scores.append(float(score))
    return scores


def accuracy_reward(completions, solution, **kwargs):
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    def _extract_answer(text):
        m = re.search(r"<SOLUTION>([\s\S]*?)</SOLUTION>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        cand = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        return cand[-1].strip() if cand else text.strip()

    for completion, sol in zip(completions, solution):
        content = _join_completion(completion)
        gt = _extract_answer(sol).replace(" ", "").replace("_", "").lower()
        pred = _extract_answer(content).replace(" ", "").replace("_", "").lower()

        if pred not in ("yes", "no"):
            reward = 0.0
        else:
            reward = 1.0 if pred == gt else 0.0

        rewards.append(float(reward))

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content(repr): {repr(content)}\n")
                f.write(f"pred: {pred} | gt: {gt}\n")
    return rewards



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <REASONING> </REASONING> and <SOLUTION> </SOLUTION> tags, respectively, i.e., "
    "<REASONING> reasoning process here </REASONING><SOLUTION> answer here </SOLUTION>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    # import pdb; pdb.set_trace()
    script_args.reward_funcs = ['accuracy','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # import pdb; pdb.set_trace()

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    ### lzy modified
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen3VLGRPOTrainer if not training_args.use_vllm else Qwen3VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
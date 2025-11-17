# ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models

This repo provides the source code & data of our paper: [**ORIC: Benchmarking Object Recognition in Incongruous Context for Large Vision-Language Models**](https://arxiv.org/abs/2509.15695) by

[Zhaoyang Li](https://zhaoyangli-1.github.io/)\* (UC San Diego), [Zhang Ling](https://lz1oceani.github.io/)\* (UC San Diego), [Yuchen Zhou](https://www.yuchenzhou.org/) (UC San Diego), [Litian Gong](https://gonglitian.github.io/) (UC Riverside), [Erdem Bıyık](https://ebiyik.github.io/) (University of Southern California), [Hao Su](https://cseweb.ucsd.edu/~haosu) (UC San Diego, Hillbot)  

\* Equal contribution.

![ORIC Overview](./figures/oric.png)


## 1. Setup:
```bash
https://github.com/ZhaoyangLi-1/ORIC.git
cd ORIC
conda create -n ORIC python=3.10
conda activate ORIC
bash setup.sh
```

## 2. Set your OpenAI API Key:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## 3. Generate ORIC QA Samples:
```bash
python main.py \
  --data_folder /path/to/coco \
  --output_folder /path/to/output \
  --num_objects 3 \
  --num_images 1000 \
  --seed 42 \
  --llm_model gpt-4o-2024-08-06 \
  --reject_prompt ./prompts/reject_sample.txt \
  --split val
```

Arguments:

--data_folder: Path to your COCO dataset folder.

--output_folder: Directory to save generated Q&A samples.

--num_objects: Number of objects to sample per image.

--num_images: Number of image pairs to use. The total number of Q&A pairs ≈ 2 × num_images × num_objects.

--llm_model: OpenAI model name (e.g., gpt-4o-2024-08-06).

--reject_prompt: Prompt template used to formulate questions.

--split: Dataset split to use: `train` or `val`.  
- `train`: generates ORIC-style training data  
- `val`: generates ORIC-Bench evaluation data


This step produces ORIC-style Q&A pairs ready for inference. We already provide generated questions in the outputs folder for dirrectly using.


## 4. Run Inference with Your VLM:

Run your Vision-Language Model (VLM) on the generated ORIC Q&A pairs. The output should be saved in a JSON file with the following structure:

```json
[
  {
    "question_id": "1",
    "predicted_answer": "yes",
    "solution": "yes"
  },
  {
    "question_id": "2",
    "predicted_answer": "no",
    "solution": "no"
  }
]
```

## 5. Evaluate Model Performance:
```bash
python evaluate.py \
  --result_path /path/to/predictions.json \
  --output_folder /path/to/eval_results
```

## 6. Visual-RFT Finetuning

This section describes how to fine-tune **Qwen3-VL-8B-Instruct** using **Visual-RFT**, our reinforcement-learning–based finetuning pipeline built upon **Group Relative Policy Optimization (GRPO)** for binary object-existence classification on ORIC-style data.

### Requirements
- 4 × NVIDIA H100 / A100 GPUs  
- PyTorch ≥ 2.1  
- Flash-Attention v2  
- DeepSpeed ZeRO-3 (config included in repo)

---

### 6.1 Training Command

Run the following command to launch GRPO fine-tuning on 4 GPUs:

```bash
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    virft/src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed virft/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 15 \
    --run_name Qwen3-VL-8B_GRPO_oric \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine
```

Arguments:

--output_dir ${SAVE_PATH}
Directory where all finetuned model weights, logs, and checkpoints will be saved.

--model_name_or_path ${CKPT_PATH}
Path to the pretrained base model (e.g., Qwen3-VL-8B-Instruct) used as the initialization for Visual-RFT.

--dataset_name ${DATA_PATH}
Path to the ORIC-style training dataset in JSON / HF Dataset format for GRPO finetuning.
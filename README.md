# ORIC
This repo provides the source code & data of our paper: ORIC: Benchmarking Object Recognition in Incongruous Context for Large Vision-Language Models


## 1. Installation Dependencies:
pip install -r requirements.txt

## 2. Set your OpenAI API Key:
export OPENAI_API_KEY="your_openai_api_key"

## 3. Generate ORIC QA Samples:
python main.py \
  --data_folder /path/to/coco \
  --output_folder /path/to/output \
  --num_objects 3 \
  --num_images 1000 \
  --seed 42 \
  --llm_model gpt-4o-2024-08-06 \
  --reject_prompt ./prompts/reject_sample.txt

Arguments:

--data_folder: Path to your COCO dataset folder.

--output_folder: Directory to save generated Q&A samples.

--num_objects: Number of objects to sample per image.

--num_images: Number of image pairs to use. The total number of Q&A pairs ≈ 2 × num_images × num_objects.

--llm_model: OpenAI model name (e.g., gpt-4o-2024-08-06).

--reject_prompt: Prompt template used to formulate questions.

This step produces ORIC-style Q&A pairs ready for inference. We already provide generated questions in the outputs folder for dirrectly using.


## 4. Run Inference with Your VLM:

Run your Vision-Language Model (VLM) on the generated ORIC Q&A pairs. The output should be saved in a JSON file with the following structure:

[
  {
    "question_id": "000001",
    "answer": "yes"
  },
  {
    "question_id": "000002",
    "answer": "no"
  }
]

## 5. Evaluate Model Performance：
python evaluate.py \
  --result_path /path/to/predictions.json \
  --output_folder /path/to/eval_results


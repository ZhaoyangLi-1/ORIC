pip uninstall -y transformers huggingface-hub
pip install "transformers>=4.51.0,<5.0" "huggingface-hub>=0.23,<1.0"
pip install -e ".[dev]"

# Additional modules
echo "Installing dependencies..."

pip install \
    pycocotools \
    shapely \
    torch \
    numpy \
    Pillow \
    tqdm \
    transformers \
    scikit-learn \
    openai \
    wandb \
    tensorboardx \
    qwen_vl_utils

pip uninstall -y torch torchvision torchaudio

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -v flash-attn --no-build-isolation




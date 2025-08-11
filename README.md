# coco-captions-extended
Using VLMs to generate synthetic captions of various lengths for COCO

## Installation

CUDA 12.0 or higher is required

```bash
python3 -m pip install --upgrade pip
python3 -m pip install wheel
python3 -m pip install ninja

python3 -m pip install torch torchvision torchaudio
python3 -m pip install git+https://github.com/huggingface/transformers accelerate
python3 -m pip install flash-attn --no-build-isolation
python3 -m pip install qwen-vl-utils
```

If you wish to use the Gemma model, you must also install the huggingface cli and register your access token. Ensure that you have access to the Gemma model on hugging face

## Running

```bash
python3 main.py --karpathy "loc/to/dataset_coco.json" \
                --coco_img_root "loc/to/coco/img/root" \
                --output "loc/to/output/json" \
                --target_seq_len 100 \
                --stats
```

source .venv/bin/activate

seq_len=100

python3 main.py --karpathy "/home/henry/Datasets/coco/dataset_coco.json" \
                --coco_img_root "/home/henry/Datasets/coco/img/" \
                --output "/home/henry/Datasets/coco/dataset_coco_ext${seq_len}.json" \
                --target_seq_len $seq_len \
                --batch_size 1 \
                # --stats

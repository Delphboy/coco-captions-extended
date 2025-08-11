import os
import argparse
import logging
import time

from typing import List

import torch

from coco import Coco, load_karpathy_split, save_karpathy_split, get_sentences, get_img_path
import models

logging.basicConfig(level=logging.DEBUG)
torch.set_float32_matmul_precision('high')

CAPTIONS_PER_IMAGE = 5

def calculate_sentence_statistics(coco: Coco):
    sentence_lengths = []

    for i, element in enumerate(coco.images):
        logging.info(f"STATS: Processing image {i}/{len(coco.images)}")
        sentence_lengths += [len(s.raw.split(' ')) for s in element.sentences]

    logging.info(f"Average Caption Length: {sum(sentence_lengths) / len(sentence_lengths)}")
    logging.info(f"Shortest Caption Length: {min(sentence_lengths)}")
    logging.info(f"Longest Caption Length: {max(sentence_lengths)}")
    logging.info("-"*10)

def opts_checker(opts):
    assert os.path.exists(opts.karpathy), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.karpathy}"
    assert os.path.exists(opts.coco_img_root), f"DIRECTORY NOT FOUND: COCO img root not found at: {opts.coco_img_root}"
    assert "train2014" in os.listdir(opts.coco_img_root), f"DIRECTORY NOT FOUND: COCO img root does not contain 'train2014'"
    assert "val2014" in os.listdir(opts.coco_img_root), f"DIRECTORY NOT FOUND: COCO img root does not contain 'val2014'"
    assert "test2014" in os.listdir(opts.coco_img_root), f"DIRECTORY NOT FOUND: COCO img root does not contain 'test2014'"

def tokeniser(caption: str) -> List[str]:
    caption = caption.lower()
    caption = caption.replace('’', "'")
    caption = caption.replace("“", ' " ')
    caption = caption.replace("”", ' " ')
    caption = caption.replace("-", ' - ') 
    caption = caption.replace("–", ' - ')
    caption = caption.replace("-", ' - ')
    caption = caption.replace(',', ' , ')
    caption = caption.replace('.', ' . ')
    caption = caption.replace('  ', ' ')
    caption_list = caption.split(' ')
    if '' in caption_list: caption_list.remove('')
    return caption_list

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--karpathy", type=str, required=True)
    args.add_argument("--coco_img_root", type=str, required=True)
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--target_seq_len", type=int, required=True)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--stats", action="store_true")

    opts = args.parse_args()

    opts_checker(opts)
    logging.info(opts)

    coco = load_karpathy_split(opts.karpathy)
    new_coco = load_karpathy_split(opts.karpathy)

    vlm = models.Gemma(opts.target_seq_len)
    # vlm = models.Qwen(opts.target_seq_len)

    for img_start_idx in range(0, len(coco.images), opts.batch_size):
        start = time.time()
        end_batch_idx = img_start_idx+opts.batch_size if img_start_idx+opts.batch_size < len(coco.images) else img_start_idx + (len(coco.images) - img_start_idx)
        indicies = [i for i in range(img_start_idx, end_batch_idx)]

        img_paths = [os.path.join(opts.coco_img_root, get_img_path(coco.images[i])) for i in indicies]
        sentences = [get_sentences(coco.images[i]) for i in indicies]
        assert len(img_paths) == len(sentences), "Different number of images to caption sets"
        assert sum([len(s) for s in sentences]) % CAPTIONS_PER_IMAGE == 0, f"Number of sentences contains an element {[len(s) for s in sentences]} that is not divisible by {CAPTIONS_PER_IMAGE}"

        new_captions = vlm.generate_caption(img_paths, sentences)
        assert len(new_captions) == len(indicies) * CAPTIONS_PER_IMAGE, f"Received {len(new_captions)} new captions but expected {len(indicies) * CAPTIONS_PER_IMAGE}"

        for i, img_idx in enumerate(indicies):
            for caption_idx in range(len(get_sentences(new_coco.images[img_idx]))):
                caption = new_captions[i * CAPTIONS_PER_IMAGE + caption_idx]
                logging.debug(f"\t{i * CAPTIONS_PER_IMAGE + caption_idx} - {caption}")

                new_coco.images[img_idx].sentences[caption_idx].raw = caption
                new_coco.images[img_idx].sentences[caption_idx].tokens = tokeniser(caption)
        end = time.time()
        logging.info(f"Processed {img_start_idx} -> {end_batch_idx}/{len(coco.images)}\t\t{end-start} seconds")

    save_karpathy_split(new_coco, opts.output)
    if opts.stats:
        calculate_sentence_statistics(new_coco)




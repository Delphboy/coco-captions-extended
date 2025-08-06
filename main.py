import os
import argparse
import logging

from typing import List

from coco import Coco, load_karpathy_split, save_karpathy_split, get_sentences, get_img_path
import models

logging.basicConfig(level=logging.INFO)

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
    caption = caption.replace(',', ' , ')
    caption = caption.replace('.', ' . ')
    caption = caption.replace('  ', ' ')
    caption_list = caption.split(' ')
    caption_list.remove('')
    return caption_list

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--karpathy", type=str, required=True)
    args.add_argument("--coco_img_root", type=str, required=True)
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--target_seq_len", type=int, required=True)
    args.add_argument("--stats", action="store_true")

    opts = args.parse_args()

    opts_checker(opts)
    logging.info(opts)

    coco = load_karpathy_split(opts.karpathy)
    new_coco = load_karpathy_split(opts.karpathy)

    if opts.stats:
        calculate_sentence_statistics(coco)

    vlm = models.Qwen(opts.target_seq_len)

    for i, coco_element in enumerate(coco.images):
        logging.info(f"Processing {i+1}/{len(coco.images)}")

        img_path = os.path.join(opts.coco_img_root, get_img_path(coco_element))
        sentences = get_sentences(coco_element)

        for s_i, sentence in enumerate(sentences):
            logging.info(f"\tCaption {s_i+1}/5")
            cap = vlm.generate_caption(img_path, sentence)
            new_coco.images[i].sentences[s_i].raw = cap
            new_coco.images[i].sentences[s_i].tokens = tokeniser(cap)

        if i == 0:
            break

    save_karpathy_split(new_coco, opts.output)
    if opts.stats:
        calculate_sentence_statistics(new_coco)




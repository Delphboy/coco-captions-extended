import os
import argparse
import logging

from coco import Coco, CocoElement, load_karpathy_split, save_karpathy_split

logging.basicConfig(level=logging.INFO)

def opts_checker(opts):
    assert os.path.exists(opts.train_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.train_split}"
    assert os.path.exists(opts.restval_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.restval_split}"
    assert os.path.exists(opts.val_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.val_split}"
    assert os.path.exists(opts.test_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.test_split}"


def get_element_from_coco_id(coco: Coco, coco_id: int) -> CocoElement:
    for element in coco.images:
        if element.cocoid == coco_id:
            return element
    raise Exception(f"ELEMENT NOT FOUND: cocoid={coco_id} is not found in the Coco object provided")


def process(new_element: CocoElement, original: Coco) -> CocoElement:
    global total_sentence_swaps
    og_element = get_element_from_coco_id(original, new_element.cocoid)
    og_sentences = og_element.sentences
    for i, new_sentence in enumerate(new_element.sentences):
        if len(new_sentence.tokens) < len(og_sentences[i].tokens):
            total_sentence_swaps += 1
            new_element.sentences[i].imgid = og_sentences[i].imgid
            new_element.sentences[i].raw = og_sentences[i].raw
            new_element.sentences[i].sentid = og_sentences[i].sentid
            new_element.sentences[i].tokens = og_sentences[i].tokens
    return new_element


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--original", type=str, required=True)
    args.add_argument("--train_split", type=str, required=True)
    args.add_argument("--restval_split", type=str, required=True)
    args.add_argument("--val_split", type=str, required=True)
    args.add_argument("--test_split", type=str, required=True)

    opts = args.parse_args()

    opts_checker(opts)
    logging.info(opts)

    original = load_karpathy_split(opts.original)
    train = load_karpathy_split(opts.train_split)
    restval = load_karpathy_split(opts.restval_split)
    val = load_karpathy_split(opts.val_split)
    test = load_karpathy_split(opts.test_split)
    new_coco = Coco(train.dataset, [])

    total_sentence_swaps = 0

    for split in [train, restval, val, test]:
        for element in split.images:
            print(f"Processing image id: {element.cocoid}")
            element = process(element, original)
            new_coco.images.append(element)

    logging.info(f"There were {total_sentence_swaps} sentence swaps out of a possible {len(new_coco.images) * 5}. i.e. {(total_sentence_swaps / float(len(new_coco.images) * 5)) * 100}%")
    save_karpathy_split(train, opts.output)




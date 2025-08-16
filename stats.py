import os
import argparse
import logging


from coco import Coco, load_karpathy_split

logging.basicConfig(level=logging.INFO)


def calculate_sentence_statistics(coco: Coco):
    sentence_lengths = []

    for i, element in enumerate(coco.images):
        logging.debug(f"STATS: Processing image {i}/{len(coco.images)}")
        sentence_lengths += [len(s.raw.split(' ')) for s in element.sentences]

    logging.info(f"Average Caption Length: {sum(sentence_lengths) / len(sentence_lengths)}")
    logging.info(f"Shortest Caption Length: {min(sentence_lengths)}")
    logging.info(f"Longest Caption Length: {max(sentence_lengths)}")
    logging.info("-"*10)


def opts_checker(opts):
    assert os.path.exists(opts.input), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.input}"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True)

    opts = args.parse_args()

    opts_checker(opts)
    logging.info(opts)

    coco = load_karpathy_split(opts.input)
    calculate_sentence_statistics(coco)




import os
import argparse
import logging

from stats import calculate_sentence_statistics

from coco import load_karpathy_split, save_karpathy_split

logging.basicConfig(level=logging.INFO)


def opts_checker(opts):
    assert os.path.exists(opts.train_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.train_split}"
    assert os.path.exists(opts.restval_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.restval_split}"
    assert os.path.exists(opts.val_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.val_split}"
    assert os.path.exists(opts.test_split), f"FILE NOT FOUND: Karpathy JSON file not found at: {opts.test_split}"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--output", type=str, required=True)
    args.add_argument("--train_split", type=str, required=True)
    args.add_argument("--restval_split", type=str, required=True)
    args.add_argument("--val_split", type=str, required=True)
    args.add_argument("--test_split", type=str, required=True)
    args.add_argument("--stats", action="store_true")

    opts = args.parse_args()

    opts_checker(opts)
    logging.info(opts)

    train = load_karpathy_split(opts.train_split)
    restval = load_karpathy_split(opts.restval_split)
    val = load_karpathy_split(opts.val_split)
    test = load_karpathy_split(opts.test_split)

    for split in [restval, val, test]:
        for element in split.images:
            train.images.append(element)


    save_karpathy_split(train, opts.output)
    if opts.stats:
        calculate_sentence_statistics(train)




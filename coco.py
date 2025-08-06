from typing import List
from dataclasses import dataclass, asdict

import os
import json

from PIL import Image

@dataclass
class Sentences:
    imgid: int
    raw: str
    sentid: int
    tokens: List[str]


@dataclass
class CocoElement:
    cocoid: int
    filename: str
    filepath: str
    imgid: int
    sentences: List[Sentences]
    sentids: List[int]
    split: str

@dataclass
class Coco:
    dataset: str
    images: List[CocoElement]

def get_sentences(coco_element: CocoElement) -> List[str]: return [s.raw for s in coco_element.sentences]
def get_img_path(coco_element: CocoElement) -> str: return os.path.join(coco_element.filepath, coco_element.filename)
def get_img(path: str) -> Image.Image: return Image.open(path).convert("RGB")

def load_karpathy_split(karpathy_file_dir: str) -> Coco:
    with open(karpathy_file_dir, 'r') as f:
        data = json.load(f)

    images = []

    for image_data in data['images']:
        sentences = [
            Sentences(
                imgid=image_data['imgid'],
                raw=sentence['raw'],
                sentid=sentence['sentid'],
                tokens=sentence['tokens']
            ) for sentence in image_data['sentences']
        ]

        coco_element = CocoElement(
            cocoid=image_data['cocoid'],
            filename=image_data['filename'],
            filepath=image_data['filepath'],
            imgid=image_data['imgid'],
            sentences=sentences,
            sentids=image_data['sentids'],
            split=image_data['split']
        )

        images.append(coco_element)

    return Coco(
        dataset=data['dataset'],
        images=images
    )

def save_karpathy_split(coco_object: Coco, karpathy_file_dir:str) -> None:
    coco_dict = asdict(coco_object)
    with open(karpathy_file_dir, 'w') as f:
        json.dump(coco_dict, f, indent=2)


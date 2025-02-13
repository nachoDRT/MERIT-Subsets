from datasets import load_dataset, Image
from os.path import join, abspath, dirname
import os
import json
from typing import Dict
from io import BytesIO
import base64


def get_dataset_iterator(subset_name: str, decode=None):

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", subset_name, split="test", streaming=True)

    if decode:
        dataset = dataset.cast_column("image", Image(decode=False))

    dataset_iterator = iter(dataset)

    return dataset_iterator


def save_dataset_jsonl(file_name, dataset_jsonl):

    path = join(dirname(dirname(abspath(__file__))), "output", file_name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for item in dataset_jsonl:
            f.write(json.dumps(item) + "\n")

    # Return the file as an object
    return open(path, "r")


def load_secrets(file_path: str) -> Dict:

    with open(file_path, encoding="utf-8") as config_file:
        secrets = json.load(config_file)

    return secrets


def encode_image(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def get_sample_data(sample):

    print("Getting Image")
    img = sample["image"]
    gt = json.loads(sample["ground_truth"])
    gt = gt["gt_parse"]

    return img, gt


def get_sample_img_name(sample):

    print("Getting Image Name")
    img_name = sample["image"]["path"]

    return img_name

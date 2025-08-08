from typing import List

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Gemma:
    def __init__(self, target_seq_len:int=150):
        self.model_name = "google/gemma-3-4b-it"
        self.target_seq_len = target_seq_len
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_name, padding=True, padding_side="left")

    def generate_caption(self, image_dirs: List[str], sentences: List[List[str]]):
        messages = []

        for i in range(len(image_dirs)):
            for j in range(len(sentences[0])):
                prompt = f"""Looking at the image provided, your task is to generate a description of the image that is between {int(0.8 * self.target_seq_len)} and {int(1.2 * self.target_seq_len)} words long. A human annotator described the image as:'{sentences[i][j]}'. Ensure that your caption is in English and imitates the style of the human generated caption"""

                msg = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_dirs[i]
                         },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
                messages.append(msg)

        assert len(messages) == len(image_dirs) * len(sentences[0]), f"Message length ({len(messages)}) is unexpected - expected {len(image_dirs) * len(sentences[0])}"

        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        input_len = inputs["input_ids"].shape[-1]

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.target_seq_len * 2)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return [ot.split(":")[1].replace("\n", "") for ot in output_texts]


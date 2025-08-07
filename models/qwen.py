from typing import List

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, GenerationConfig, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Qwen:
    def __init__(self, target_seq_len:int=150):
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            # torch_dtype="auto", 
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        ).eval()

        self.target_seq_len = target_seq_len

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_name, 
                                                       use_fast=False, 
                                                       padding_side='left', 
                                                       min_pixels=min_pixels, 
                                                       max_pixels=max_pixels)

        self.generation_config = GenerationConfig.from_pretrained(self.model_name,
                                                                  max_length=1.2*target_seq_len, 
                                                                  min_length=0.8*target_seq_len)

    def generate_caption(self, image_dirs: List[str], sentences: List[List[str]]):
        messages = []

        for i in range(len(image_dirs)):
            for j in range(len(sentences[0])):
                prompt = f"""Looking at the image provided, your task is to generate a description of the image that is between {int(0.8 * self.target_seq_len)} and {int(1.2 * self.target_seq_len)} words long. A human annotator described the image as:'{sentences[i][j]}'. Ensure that your caption is in English and imitates the style of the human generated caption. Your description is being used to replace the example given so do not include anything other than the description. Your output is being copied directly"""

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

        prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)

        assert len(prompts) == len(image_inputs), "Different number of images and prompts. Should be equal"

        inputs = self.processor(text=prompts, 
                                images=image_inputs, 
                                videos=video_inputs, 
                                padding=True,
                                return_attention_mask=True,
                                return_tensors="pt")
        inputs = inputs.to(DEVICE)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.target_seq_len * 2)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return [ot.replace("\n", "") for ot in output_texts]

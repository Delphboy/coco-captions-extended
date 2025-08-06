from transformers import AutoProcessor
import torch
from transformers.image_utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Qwen:
    def __init__(self, target_seq_len:int=150):
        self.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.target_seq_len = target_seq_len
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": ""
                },
                {
                    "type": "text", 
                    "text": ""
                }
            ]
        }]

        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(max_length=1.2*target_seq_len, min_length=0.8*target_seq_len)

    def generate_caption(self, image_dir, caption):
        image = load_image(image_dir)

        self.messages[0]["content"][0]["image"] = image_dir

        self.messages[0]["content"][1]["text"] = f"""
        Looking at the image provided, your task is to generate a description of the image that is between {0.8 * self.target_seq_len} and {1.2 * self.target_seq_len} words long.
        Try to imitate the style of the following caption: '{caption}'
        Ensure that the caption is in English
        """

        prompt = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(text=[prompt], 
                                images=image_inputs, 
                                videos=video_inputs, 
                                padding=True,
                                return_tensors="pt")
        inputs = inputs.to(DEVICE)


        generated_ids = self.model.generate(**inputs, max_new_tokens=self.target_seq_len * 2)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].replace("\n", "").replace("\r", "")


from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import torch
from PIL import Image
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SmolVLM:
    def __init__(self, target_seq_len:int=150):
        self.model_name = "HuggingFaceTB/SmolVLM-Instruct"
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, 
                                                                 torch_dtype=torch.bfloat16, 
                                                                 _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager").to(DEVICE)
        self.target_seq_len = target_seq_len
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": ""
                }
            ]
        }]

    def generate_caption(self, image_dir, caption):
        image = load_image(image_dir)

        self.messages[0]["content"][1]["text"] = f"The given image was described by a human annotator as the following: '{caption}'. Your task is to generate a longer description of this image of {self.target_seq_len} words. Do not generate anything other than the description."

        prompt = self.processor.apply_chat_template(self.messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]


import sys
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

from utils import add_white_borders, get_image_path_from_cli

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

def set_pipe(model_name="runwayml/stable-diffusion-inpainting"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_name,
    revision="fp16",
    torch_dtype=torch.float16,
    ).to('cuda')
    pipe.safety_checker=None
    return pipe

if __name__=='__main__':

    image_path = get_image_path_from_cli()
    img = Image.open(image_path).convert('RGB')
    
    zoomed_out_image, mask, border_size = add_white_borders(np.array(img))

    # unconditional image captioning
    text = "a photograph of"
    inputs = blip_processor(img, text, return_tensors="pt").to("cuda", torch.float16)

    out = blip_model.generate(**inputs, max_new_tokens=20)
    prompt = blip_processor.decode(out[0], skip_special_tokens=True)
    print(prompt)
    pipe = set_pipe()
    
    out = pipe(
        image=Image.fromarray(zoomed_out_image), 
        prompt = prompt,
        mask_image=Image.fromarray(mask), 
        num_inference_steps=20,
        num_images_per_prompt=1
            )
    out = out[0][0].resize(Image.fromarray(zoomed_out_image).size)
    out.save('images/output.png')
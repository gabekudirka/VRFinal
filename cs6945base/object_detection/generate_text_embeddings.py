from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, CLIPTextModel
import torch
from PIL import Image
import json

if __name__ == '__main__':
    captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    captioning_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captioning_model.to(device)

    def predict_step(image_paths, gen_kwargs):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = captioning_model.generate(pixel_values, **gen_kwargs)

        preds = captioning_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    with open('./Meta-DETR/data/voc_fewshot_split1/seed1/3shot.json') as f:
        data_3shot = json.load(f)
    image_paths = [image.file_name for image in data_3shot['images']]

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    desc = predict_step(image_paths, gen_kwargs)

    clip_embedding_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    embedding_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer(desc, padding=True, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output 
    print(len(pooled_output))
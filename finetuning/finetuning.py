from transformers import LLAVAModel

model = LLAVAModel.from_pretrained("llava-base")
print("Model loaded successfully!")
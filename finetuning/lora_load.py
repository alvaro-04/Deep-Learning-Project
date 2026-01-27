from unsloth import FastVisionModel


model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_llava",
    load_in_4bit = False # CHANGE IF WE USE 4BIT
)
FastVisionModel.for_inference(model)




#TODO: load from dataset as image and input_text
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
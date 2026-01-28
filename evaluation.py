import re
import torch
from datasets import load_from_disk
from unsloth import FastVisionModel


choices_re = re.compile(r"\b([ABCD])\b", re.I)

def extract_abcd(text: str) -> str:
    answer = choices_re.search((text or "").strip())
    if answer:
        return answer.group(1).upper()
    return ""

def main():
    with torch.inference_mode():
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name="lora_llava",
            load_in_4bit=False, 
        )
        FastVisionModel.for_inference(model)

        device = "cuda"
        model.to(device)

        ds = load_from_disk("./training_dataset/image_test")

        correct = 0
        total = 0

        # for testing using first 500
        for i in range(500):
            sample = ds[i]
            image = sample["image"]
            image = image.convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]},
                ],
            }]

            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)

            out_token_ids= model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            )

            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = out_token_ids[0, prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)


            predict_answer = extract_abcd(gen_text)
            answer = str(sample["answer"]).strip().upper()

            total += 1
            correct += int(predict_answer == answer)

            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(ds)}] acc so far: {correct/total:.4f}")

        print("\nFinal accuracy:", correct / total)

if __name__ == "__main__":
    main()


import torch
from datasets import load_dataset
from transformers import (
    LlavaForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model #parameter efficient fine tuning

from trl import SFTTrainer # easier training w/ trl library

# using 4-bit quantization
model_id = "llava-hf/llava-1.5-7b-hf"
# model_id = "llava-hf/llava-v1.6 -mistral-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target the language model layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


#placeholder for preprocessing
def format_data():
    pass

dataset = load_dataset("csv", data_files="train.csv")
dataset = dataset.map(format_data)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./llava-pmcvqa-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True, # Recommended for newer GPUs
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    report_to="none" 
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text", # The field we created in format_data
    max_seq_length=512,
    dataset_kwargs={"skip_prepare_dataset": True} # Processor handles multimodal data
)

trainer.train()
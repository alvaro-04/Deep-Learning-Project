from unsloth import FastVisionModel
from datasets import load_dataset, Image
import torch
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import os


# Import model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
    load_in_4bit = False, # 16 bit lora, if OOM switch to true
    use_gradient_checkpointing = "unsloth" # may help batching
)

# Import PEFT
model = FastVisionModel.get_peft_model(
    model,
    # Finetune on every single layer
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 

    r = 16,           
    lora_alpha = 16,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None,
)

# Memory tracking
start_gpu_memory = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)

# Getting data from parquet
dataset = load_dataset("parquet", data_files={
    'train': 'training_dataset/train.parquet', 
    'test': 'training_dataset/test.parquet'
})['train']

# from llava 1.5 kaggle notebook  https://www.kaggle.com/code/sachidanandnavik/fine-tuning-llava-1-5-7b-on-agricultural-datasets
def transform_to_llava_format(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{sample['question']}\nAnswer with the letter A, B, C or D."},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]}
                ]
            }
        ]
    }

# Apply the transformation
dataset = dataset.map(
    transform_to_llava_format, 
    remove_columns=['question', 'answer', 'image'],  # Remove original columns
    num_proc=os.cpu_count()
)

###########Training step#################

FastVisionModel.for_training(model)

# Configure trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 1000,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",

        remove_unused_columns = False,
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = 2048,
    ),
)

# Begin training
trainer_stats = trainer.train()

# Post-training stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


##########MODEL SAVING#############

# NB: only saves LoRA heads!!
model.save_pretrained("lora_llava")
tokenizer.save_pretrained("lora_llava")
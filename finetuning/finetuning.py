from unsloth import FastVisionModel
from datasets import load_dataset
import torch
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


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

#Getting data (assuming in parquet format as a placeholder for now)
dataset = load_dataset("parquet", data_files={
    'train': 'training_dataset/train.parquet', 
    'test': 'training_dataset/test.parquet'
})
instruction = "Answer the following question with either A, B, C or D."

#Assuming parquet consists of image, question and response
def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text", "text"  : instruction},
            {"type" : "image", "image" : sample["image"]},
            {"type" : "text", "text" : sample["question"]}]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["response"]} ]
        },
    ]
    return { "messages" : conversation }

# Convert to format
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

###########Training step#################

FastVisionModel.for_training(model)

# Configure trainer
# (Parameters from Unsloth docs)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

#Begin training
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

#NB: only saves LoRA heads!!
model.save_pretrained("lora_llava")
tokenizer.save_pretrained("lora_llava")
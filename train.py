"""
Generic trainer script to fine tune LLMs in an alpaca formatted dataset

Based on Abhishek Thakur video "Train LLMs in just 50 lines of code!"

Check README.md for more details
"""
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


INPUT_DATASET = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
DATA_TXT_FIELD =  "text"
MAX_SEQUENCE_LEN = 1024
USE_FP_16 = False # True # Only use in cuda. TPUs and M1 mac GPU needs it False
MODEL = "salesforce/xgen-7b-8k-base"
OUTPUT_DIR = "xgen-7b-8k-base"


def train():
    train_dataset = load_dataset(INPUT_DATASET,
                                 split="train",
                                 )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                              trust_remote_code=True,
                                              )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                 trust_remote_code=True,
                                                 load_in_4bit=True,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto",
                                                 )
    
    # not required for most recent models but it is good to do it anyway
    model.resize_token_embeddings(len(tokenizer))
    
    model = prepare_model_for_int8_training(model)

    peft_config = LoraConfig(r=16,
                             lora_alpha=32,
                             lora_dropout=0.05,
                             bias="none",
                             task_type="CAUSAL_LM",
                             )
    
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(output_dir=OUTPUT_DIR,
                                      per_device_train_batch_size=1,
                                      gradient_accumulation_steps=8,
                                      optim="adamw_torch",
                                      logging_steps=100,
                                      learning_rate=2e-4,
                                      fp16=USE_FP_16,
                                      warmup_ratio=0.1,
                                      lr_scheduler_type="linear",
                                      num_train_epochs=1,
                                      save_strategy="epoch",
                                      push_to_hub=False,
                                      )
    
    trainer = SFTTrainer(model=model,
                         train_dataset=train_dataset,
                         dataset_text_field=DATA_TXT_FIELD,
                         max_seq_length=MAX_SEQUENCE_LEN,
                         tokenizer=tokenizer,
                         args=training_args,
                         peft_config=peft_config,
                         packing=True,
                         )
    
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()

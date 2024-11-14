import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig

def main(dataset_path: str, model_name: str, load_in_4_bit: bool = False, train_num: int | None = 1000, eval_num: int | None = 20):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    if load_in_4_bit: 
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=nf4_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # 配置 LoRA
    lora_config = LoraConfig(
        r=128,  # LoRA 低秩矩阵的秩
        lora_alpha=32,  # LoRA 超参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,  # LoRA dropout
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    from benchmark_test import NLDataset
    train_dataset = NLDataset(path=os.path.join(dataset_path, "train_rfft.json"), train_or_test="train", num_each=train_num, tokenizer=tokenizer)
    eval_dataset = NLDataset(path=os.path.join(dataset_path, "valid_rfft.json"), train_or_test="train", num_each=eval_num, tokenizer=tokenizer)

    collator = DataCollatorForCompletionOnlyLM(response_template=[567, 6075, 512], tokenizer=tokenizer, mlm=False) # 567, 6075, 512 are the indices of the tokens in llama3.1


    # 设置训练参数
    training_args = SFTConfig(
        output_dir="./rfft/finetuned",
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=300,
        logging_dir="./logs_2",
        logging_steps=100,
        max_seq_length=2048,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        load_best_model_at_end=True,
    )

    # 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    trainer.train()
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--load_in_4_bit", action="store_true")
    parser.add_argument("--train_num", type=int, default=1000)
    parser.add_argument("--eval_num", type=int, default=20)
    args = parser.parse_args()
    
    main(args.dataset_path, args.model_name, args.load_in_4_bit, args.train_num, args.eval_num)
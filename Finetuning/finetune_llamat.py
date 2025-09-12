#!/usr/bin/env python3
"""
HPC-Compatible LLaMat Fine-tuning
Fine-tunes the pre-trained LLaMat model for carbon aerogel research
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import gc

# Enable better CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

class LLaMatFineTuner:
    def __init__(self, model_path="m3rg-iitd/llamat-2", dataset_path="carbon_aerogel_llamat_mixed.json"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = "llamat_finetuned_complete"
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024"
    
    def load_model_and_tokenizer(self):
        print(f"Loading LLaMat model and tokenizer from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token correctly for LLaMat
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"✅ Pad token ID: {self.tokenizer.pad_token_id}")
        print(f"✅ EOS token ID: {self.tokenizer.eos_token_id}")
        
        # Load model with HPC-friendly settings
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"⚠️ Loading with quantization failed: {e}")
            print("🔄 Trying without quantization...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        print(f"✅ Model type: {getattr(self.model.config, 'model_type', 'unknown')}")
        print(f"✅ Model vocab size: {self.model.config.vocab_size}")
        print("✅ LLaMat model loaded successfully")
    
    def validate_and_clean_dataset(self, data):
        """Validate dataset for LLaMat"""
        print("Validating dataset for LLaMat...")
        
        valid_entries = []
        for i, item in enumerate(data):
            try:
                instruction = item.get('instruction', '').strip()
                output = item.get('output', '').strip()
                
                if not instruction or not output:
                    continue
                
                # Use LLaMat chat format
                text = f"<|system|>\nYou are L.A.R.A (LLMs as Aerogel Research Assistants), an expert materials scientist specializing in carbon aerogels.\n<|user|>\n{instruction}\n<|assistant|>\n{output}"
                
                # Test tokenization
                tokens = self.tokenizer(
                    text,
                    max_length=1024,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                # Validate token IDs
                max_token_id = max(tokens['input_ids']) if tokens['input_ids'] else 0
                if max_token_id >= self.tokenizer.vocab_size:
                    print(f"⚠️ Skipping entry {i}: token {max_token_id} >= vocab size {self.tokenizer.vocab_size}")
                    continue
                
                if any(token_id < 0 for token_id in tokens['input_ids']):
                    print(f"⚠️ Skipping entry {i}: negative token ID found")
                    continue
                
                valid_entries.append({'instruction': instruction, 'output': output})
                
            except Exception as e:
                print(f"⚠️ Error processing entry {i}: {e}")
                continue
        
        print(f"✅ Validated: {len(valid_entries)}/{len(data)} entries")
        return valid_entries
    
    def prepare_dataset(self):
        print("Preparing dataset for LLaMat...")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        valid_data = self.validate_and_clean_dataset(data)
        
        if len(valid_data) == 0:
            raise ValueError("No valid entries found in dataset!")
        
        # Format for LLaMat
        def format_example(example):
            text = f"""<|system|>
You are L.A.R.A (LLMs as Aerogel Research Assistants), an expert materials scientist specializing in carbon aerogels. Provide detailed, scientific answers about carbon aerogel synthesis, properties, and applications.
<|user|>
{example['instruction']}
<|assistant|>
{example['output']}"""
            return {"text": text}
        
        raw_dataset = Dataset.from_list(valid_data)
        self.dataset = raw_dataset.map(
            format_example,
            remove_columns=raw_dataset.column_names,
            desc="Formatting for LLaMat"
        )
        
        print(f"✅ Dataset prepared: {len(self.dataset)} examples")
    
    def setup_lora(self):
        print("Setting up LoRA for LLaMat...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LLaMat LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    
    def train_model(self):
        print("Starting LLaMat fine-tuning...")
        
        # HPC-optimized training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            optim="adamw_torch",
            num_train_epochs=3,
            learning_rate=2e-5,
            fp16=False,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            max_grad_norm=0.3,
            warmup_steps=100,
            save_steps=100,
            logging_steps=10,
            save_total_limit=2,
            dataloader_drop_last=True,
            remove_unused_columns=True,
            report_to=None,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Important for HPC
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            args=training_args,
            packing=False,
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print("🚀 Starting training...")
            trainer.train()
            trainer.save_model(self.output_dir)
            print(f"✅ Training completed - saved to {self.output_dir}")
            return trainer
        except Exception as e:
            print(f"❌ Training failed: {e}")
            try:
                trainer.save_model(f"{self.output_dir}_partial")
                print(f"✅ Partial model saved")
            except:
                pass
            raise
    
    def run_fine_tuning(self):
        print("=" * 60)
        print("HPC LLaMat Fine-Tuning for Carbon Aerogel Research")
        print("=" * 60)
        
        try:
            self.load_model_and_tokenizer()
            self.prepare_dataset()
            self.setup_lora()
            self.train_model()
            print("🎉 LLaMat fine-tuning completed!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    MODEL_NAME = "m3rg-iitd/llamat-2"
    DATASET_PATH = "carbon_aerogel_llamat_mixed.json"
    
    try:
        print("🚀 HPC LLaMat Fine-tuning for Carbon Aerogel Research")
        print(f"📦 Model: {MODEL_NAME}")
        print(f"📄 Dataset: {DATASET_PATH}")
        
        # Set HPC-friendly environment
        os.environ["HF_HUB_OFFLINE"] = "0"  # Ensure online mode
        
        # Check dataset exists
        if not os.path.exists(DATASET_PATH):
            raise SystemExit(f"Dataset not found: {DATASET_PATH}")
        
        # Run fine-tuning directly from HuggingFace Hub
        fine_tuner = LLaMatFineTuner(MODEL_NAME, DATASET_PATH)
        fine_tuner.run_fine_tuning()
        
        print(f"\n🎉 SUCCESS: LLaMat fine-tuning completed!")
        print(f"📁 Fine-tuned model: {fine_tuner.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        print("\n💡 HPC troubleshooting:")
        print("1. Ensure you're on a compute node with internet access")
        print("2. Check: module load python pytorch")
        print("3. Set cache: export HF_HOME=/path/to/cache")
        print("4. Try: salloc --gres=gpu:1 --time=4:00:00")

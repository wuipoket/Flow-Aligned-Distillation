# finetune_lora.py
# -*- coding: utf-8 -*-
import ast
import os
import sys
import math
from typing import Optional, List

import fire
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, set_seed

from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM
from transformers import TrainerCallback
import torch


HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def sanitize_hf_token_env() -> Optional[str]:
    """Keep only an ASCII-safe HF token env so hub/httpx headers don't crash."""
    resolved: Optional[str] = None
    source: Optional[str] = None

    for env_name in HF_TOKEN_ENV_NAMES:
        raw = os.environ.get(env_name, "")
        token = raw.strip()
        if not token:
            continue
        if all(ord(ch) < 128 for ch in token):
            resolved = token
            source = env_name
            break
        print(
            f"[LoRA][Warn] Ignoring non-ASCII Hugging Face token from {env_name}.",
            file=sys.stderr,
        )

    for env_name in HF_TOKEN_ENV_NAMES:
        os.environ.pop(env_name, None)

    if resolved:
        os.environ["HF_TOKEN"] = resolved
        os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved
        print(f"[LoRA] Using Hugging Face token from {source}.")
    else:
        print("[LoRA] No usable Hugging Face token found in environment.")

    return resolved


def normalize_target_modules(target_modules):
    """Accept list input, JSON/Python-style list strings, or comma-separated names."""
    if target_modules is None:
        return None
    if isinstance(target_modules, list):
        return [str(item).strip() for item in target_modules if str(item).strip()]
    if isinstance(target_modules, tuple):
        return [str(item).strip() for item in target_modules if str(item).strip()]
    if not isinstance(target_modules, str):
        return [str(target_modules).strip()]

    text = target_modules.strip()
    if not text:
        return None

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(
                f"Unable to parse target_modules={target_modules!r} as a list literal."
            ) from exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"target_modules must parse to a list/tuple, got {type(parsed).__name__}.")
        return [str(item).strip() for item in parsed if str(item).strip()]

    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]

    return text


def generate_prompt(dp):
    if dp.get("input"):
        return (f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{dp['instruction']}\n\n"
                f"### Input:\n{dp['input']}\n\n"
                f"### Response:\n{dp['output']}")
    else:
        return (f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{dp['instruction']}\n\n"
                f"### Response:\n{dp['output']}")

def tokenize_fn(tokenizer, prompt, cutoff_len, add_eos_token=True):
    res = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if add_eos_token and res["input_ids"][-1] != tokenizer.eos_token_id and len(res["input_ids"]) < cutoff_len:
        res["input_ids"].append(tokenizer.eos_token_id)
        res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    return res

def generate_and_tokenize_prompt(dp, tokenizer, cutoff_len, train_on_inputs=True):
    full = generate_prompt(dp)
    tok = tokenize_fn(tokenizer, full, cutoff_len, add_eos_token=True)
    if not train_on_inputs:
        user_prompt = generate_prompt({**dp, "output": ""})
        tok_user = tokenize_fn(tokenizer, user_prompt, cutoff_len, add_eos_token=False)
        user_len = len(tok_user["input_ids"])
        tok["labels"] = [-100] * user_len + tok["labels"][user_len:]
    return tok

def train(
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "./lora_out",
    adapter_name: str = "lora",    # (kept for CLI compatibility)
    load_8bit: bool = False,
    trust_remote_code: bool = True,
    # training
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    eval_step: int = 200,
    save_step: int = 200,
    # lora
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    # resume / other
    resume_from_checkpoint: Optional[str] = None,
    seed: int = 42,
    train_on_inputs: bool = True,
    merge_after_train: bool = False,
    merged_output_dir: str = "",
    merge_dtype: str = "bf16",
    merge_device_map: str = "auto",
):
    assert base_model, "Please specify --base_model"
    assert data_path, "Please specify --data_path"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    ddp_enabled = world_size > 1
    if ddp_enabled and torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    set_seed(seed)

    # load data
    if data_path.endswith(".json"):
        raw = load_dataset("json", data_files=data_path)
    else:
        raw = load_dataset(data_path)

    hf_token = sanitize_hf_token_env()
    target_modules = normalize_target_modules(target_modules)
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    if not target_modules:
        raise ValueError("target_modules resolved to empty; please provide at least one module name.")
    print(f"[LoRA] normalized target_modules={target_modules}")

    # load model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    model_load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": trust_remote_code,
    }
    if ddp_enabled:
        if load_8bit and local_rank >= 0:
            model_load_kwargs["device_map"] = {"": local_rank}
    else:
        model_load_kwargs["device_map"] = "auto"
    if load_8bit:
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if hf_token:
        model_load_kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        **model_load_kwargs,
    )


    # tokenizer
    tokenizer_load_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if hf_token:
        tokenizer_load_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # prepare 8bit if asked (using repo helper)
    if load_8bit:
        try:
            from peft import prepare_model_for_int8_training
            model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)
        except Exception:
            print("prepare_model_for_int8_training not found or failed - ensure peft provides it and bitsandbytes is installed.")

    # create LoRA adapter (wrap model)
    if "lora" in adapter_name.lower():
        lcfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(
            model,
            lcfg,
            autocast_adapter_dtype=False,
        )
        print("LoRA wrapper created. Trainable params (should be LoRA params only).")
                # --- 在載入 model 後（以及在 inject LoRA wrapper 後）確保關閉 cache 與正確啟動 checkpointing ---
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        # 如果你要用 gradient checkpointing（可選），請啟用並確保 inputs 能 require grads
        # NOTE: 若不想用 checkpointing，可註解掉下面兩行
        model.gradient_checkpointing_enable()
        # 這行很重要：讓 forward 的 inputs 建立可 backward 的 graph（避免 "None of the inputs have requires_grad=True"）
        try:
            model.enable_input_require_grads()
        except Exception:
            # fallback：若沒有此 API，嘗試讓 embedding 的 requires_grad=True
            if hasattr(model, "get_input_embeddings"):
                model.get_input_embeddings().weight.requires_grad_(True)

    # === 自動偵測最新 checkpoint ===
    latest_ckpt = None
    if os.path.isdir(output_dir):
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
            latest_ckpt = os.path.join(output_dir, ckpts[-1])
            print(f"[Auto-Resume] Found latest checkpoint: {latest_ckpt}")

    # 如果 CLI 有輸入 resume_from_checkpoint 就用 CLI 的，否則用自動偵測的
    if resume_from_checkpoint:
        trainer_resume_arg = resume_from_checkpoint
    else:
        trainer_resume_arg = latest_ckpt

    # 只在 CLI 或 auto-resume 都沒指定時才去找 adapter-only
    if trainer_resume_arg and os.path.isdir(trainer_resume_arg):
        # 檢查是不是完整 HF checkpoint（有 trainer_state.json）
        if os.path.exists(os.path.join(trainer_resume_arg, "trainer_state.json")):
            print(f"[Resume] Using full HF checkpoint: {trainer_resume_arg}")
        else:
            # adapter-only 模式
            cand = [
                os.path.join(trainer_resume_arg, "adapter_model.bin"),
                os.path.join(trainer_resume_arg, "pytorch_model.bin"),
                os.path.join(trainer_resume_arg, "model.bin"),
            ]
            ck = next((p for p in cand if os.path.exists(p)), None)
            if ck is not None:
                print(f"[Resume] Loading adapter from {ck}")
                state = torch.load(ck, map_location="cpu")
                model = set_peft_model_state_dict(model, state)
                trainer_resume_arg = None
            else:
                print(f"[Resume] No adapter checkpoint found under {trainer_resume_arg}")
                trainer_resume_arg = None


    # freeze everything except adapters (get_peft_model typically does this, but enforce)
    for n, p in model.named_parameters():
        if "lora_" in n or "bias" in n and "lora" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    def get_trainable_count(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("Trainable params:", get_trainable_count(model))

    # prepare datasets
    def map_fn(ex):
        return generate_and_tokenize_prompt(ex, tokenizer=tokenizer, cutoff_len=cutoff_len, train_on_inputs=train_on_inputs)

    remove_cols = raw["train"].column_names
    if val_set_size > 0:
        split = raw["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
        train_data = split["train"].shuffle(seed=seed).map(map_fn, remove_columns=remove_cols)
        val_data = split["test"].shuffle(seed=seed).map(map_fn, remove_columns=remove_cols)
    else:
        train_data = raw["train"].shuffle(seed=seed).map(map_fn, remove_columns=remove_cols)
        val_data = None

    # training args
    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    global_effective_batch_size = effective_batch_size * max(1, world_size)
    num_update_steps_per_epoch = max(1, math.ceil(len(train_data) / global_effective_batch_size))
    max_train_steps = max(1, math.ceil(num_epochs * num_update_steps_per_epoch))
    warmup_steps = max(1, math.ceil(0.03 * max_train_steps))
    print(f"[TrainArgs] world_size={world_size}, max_train_steps={max_train_steps}, warmup_steps={warmup_steps}")

    training_args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=50,
        save_strategy="steps",
        save_steps=save_step,
        eval_steps=eval_step,
        eval_strategy="steps" if val_data is not None else "no",
        save_total_limit=3,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        load_best_model_at_end=True,     # Early Stopping 需要
        metric_for_best_model="eval_loss",  # 以 eval_loss 判斷最佳
        greater_is_better=False,
    )
    if ddp_enabled:
        training_args_kwargs["ddp_find_unused_parameters"] = False
    training_args = transformers.TrainingArguments(**training_args_kwargs)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    # run training (HF Trainer supports resume_from_checkpoint if trainer_resume_arg set)
    trainer.train(resume_from_checkpoint=trainer_resume_arg)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    is_world_process_zero = trainer.is_world_process_zero()
    # 畫 loss 曲線
    import matplotlib.pyplot as plt

    if is_world_process_zero and hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        train_loss = []
        eval_loss = []
        steps = []

        for log in trainer.state.log_history:
            if "loss" in log:
                train_loss.append(log["loss"])
                steps.append(log["step"])
            if "eval_loss" in log:
                eval_loss.append((log["step"], log["eval_loss"]))

        plt.figure(figsize=(8, 5))
        plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
        if eval_loss:
            eval_steps, eval_vals = zip(*eval_loss)
            plt.plot(eval_steps, eval_vals, label="Validation Loss")

        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        loss_plot_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss curve saved to {loss_plot_path}")

    # save adapter only (PEFT wrapper handles saving adapter weights)
    if is_world_process_zero:
        os.makedirs(output_dir, exist_ok=True)
        print("Saving adapter to", output_dir)
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Done. Adapter and tokenizer saved to:", output_dir)

    if not merge_after_train:
        return
    if not is_world_process_zero:
        return

    merge_target_dir = merged_output_dir.strip() if merged_output_dir else os.path.join(output_dir, "merged")
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    merge_dtype_key = str(merge_dtype).strip().lower()
    merge_torch_dtype = dtype_map.get(merge_dtype_key, torch.bfloat16)
    merge_device_map = str(merge_device_map).strip() or "auto"

    print(f"[MERGE] start merge adapter -> full model at {merge_target_dir}")
    try:
        del trainer
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_for_merge = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=merge_torch_dtype,
        device_map=merge_device_map,
        trust_remote_code=trust_remote_code,
    )
    peft_for_merge = PeftModel.from_pretrained(
        base_for_merge,
        output_dir,
        is_trainable=False,
    )
    merged_model = peft_for_merge.merge_and_unload()
    os.makedirs(merge_target_dir, exist_ok=True)
    merged_model.save_pretrained(merge_target_dir)
    tokenizer.save_pretrained(merge_target_dir)
    print(f"[MERGE] merged full model saved to: {merge_target_dir}")


if __name__ == "__main__":
    fire.Fire(train)

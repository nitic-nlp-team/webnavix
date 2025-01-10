import json
import logging
from pathlib import Path
from typing import Any

import huggingface_hub
import hydra
import torch
import wandb
import wandb.util
from accelerate import Accelerator
from datasets import Dataset
from dotenv import load_dotenv
from mergoo.models.modeling_llama import LlamaForCausalLM
from omegaconf import OmegaConf
from peft import LoraConfig  # type: ignore  # noqa: PGH003
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # type: ignore  # noqa: PGH003
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from trl import SFTTrainer
from weblinx.utils import set_seed

load_dotenv()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    huggingface_hub.login(token=cfg.huggingface.token)
    wandb.login(key=cfg.wandb.key)

    set_seed(cfg.seed)

    model_save_dir = Path(cfg.model.save_dir).expanduser()
    model_save_dir.mkdir(exist_ok=True, parents=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_name,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    accelerator = Accelerator() if cfg.train.accelerate.use else None
    model = (
        LlamaForCausalLM.from_pretrained(
            cfg.model.base_name,
            device_map={"": accelerator.process_index} if accelerator is not None else "auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2" if cfg.model.use_flash_attention_2 else None,
            quantization_config=bnb_config if cfg.train.qlora.use else None,
        )
        if cfg.model.moe
        else AutoModelForCausalLM.from_pretrained(
            cfg.model.base_name,
            device_map={"": accelerator.process_index} if accelerator is not None else "auto",
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2" if cfg.model.use_flash_attention_2 else None,
            quantization_config=bnb_config if cfg.train.qlora.use else None,
        )
    )
    if cfg.model.freeze.use:
        for name, param in model.named_parameters():  # type: ignore  # noqa: PGH003
            if any(layer in name for layer in cfg.model.freeze.trainable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

    with Path.open(
        model_save_dir.joinpath(
            f"{cfg.train.split}/{cfg.train.domain if cfg.train.domain else ''}/input_records.json",
        ),
        "r",
    ) as f:
        train_input_records = json.load(f)
    with Path.open(
        model_save_dir.joinpath(
            f"{cfg.eval.split}/{cfg.eval.domain if cfg.eval.domain else ''}/input_records.json",
        ),
        "r",
    ) as f:
        eval_input_records = json.load(f)

    train_input_texts = [{"text": record["text"]} for record in train_input_records]
    eval_input_texts = [{"text": record["text"]} for record in eval_input_records]

    train(
        cfg,
        model,
        tokenizer,
        train_input_texts,
        eval_input_texts,
        model_save_dir,
    )


def train(  # noqa: PLR0913
    cfg,
    model,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    train_input_texts: list[dict[str, Any]],
    eval_input_texts: list[dict[str, Any]],
    model_save_dir: Path,
) -> None:
    peft_config = LoraConfig(
        r=cfg.train.qlora.r,
        lora_alpha=cfg.train.qlora.alpha,
        lora_dropout=cfg.train.qlora.dropout,
        bias=cfg.train.qlora.bias,
        task_type="CAUSAL_LM",
        target_modules=OmegaConf.to_container(cfg.train.qlora.target_modules, resolve=True),  # type: ignore  # noqa: PGH003
    )

    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        num_train_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size_per_device,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.scheduler,
        warmup_steps=cfg.train.warmup_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        per_device_eval_batch_size=cfg.eval.batch_size_per_device,
        eval_accumulation_steps=cfg.eval.gradient_accumulation_steps,
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        bf16=True,
        bf16_full_eval=True,
        group_by_length=True,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        report_to="wandb",
    )  # type: ignore  # noqa: PGH003

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,  # type: ignore  # noqa: PGH003
        train_dataset=Dataset.from_list(train_input_texts),
        eval_dataset=Dataset.from_list(eval_input_texts),
        max_seq_length=model.config.max_position_embeddings,
        dataset_text_field="text",
        peft_config=peft_config if cfg.train.qlora.use else None,
    )

    wandb.init(project=cfg.wandb.project, group="webnavix-llama")

    trainer.train()  # type: ignore  # noqa: PGH003

    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(model_save_dir)
    trainer.state.save_to_json(str(model_save_dir.joinpath("trainer_state.json")))

    wandb.finish()


if __name__ == "__main__":
    main()

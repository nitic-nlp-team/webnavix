import json
import logging
from pathlib import Path
from typing import Any

import huggingface_hub
import hydra
import torch
import wandb
from accelerate import Accelerator
from dotenv import load_dotenv
from mergoo.models.modeling_llama import LlamaForCausalLM
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # type: ignore  # noqa: PGH003
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    pipeline,  # type: ignore  # noqa: PGH003
)
from transformers.pipelines.pt_utils import KeyDataset
from weblinx.utils import set_seed

load_dotenv()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    huggingface_hub.login(token=cfg.huggingface.token)
    wandb.login(key=cfg.wandb.key)

    set_seed(cfg.seed)

    model_save_dir = Path(cfg.model.save_dir).expanduser()
    model_save_dir.mkdir(exist_ok=True, parents=True)
    result_dir = Path(cfg.eval.result_dir).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    load_model_name = str(model_save_dir) if cfg.eval.get("load_from_save_dir", False) is True else cfg.model.base_name

    tokenizer = AutoTokenizer.from_pretrained(
        load_model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    accelerator = Accelerator() if cfg.train.accelerate.use else None
    model = (
        LlamaForCausalLM.from_pretrained(
            load_model_name,
            device_map={"": accelerator.process_index} if accelerator is not None else "auto",
            torch_dtype=torch.bfloat16,
            rope_scaling={"type": "dynamic", "factor": 2.0} if cfg.model.use_rope else None,
            attn_implementation="flash_attention_2" if cfg.model.use_flash_attention_2 else None,
            quantization_config=bnb_config if cfg.train.qlora.use else None,
        )
        if cfg.model.moe
        else AutoModelForCausalLM.from_pretrained(
            load_model_name,
            device_map={"": accelerator.process_index} if accelerator is not None else "auto",
            torch_dtype=torch.bfloat16,
            rope_scaling={"type": "dynamic", "factor": 2.0} if cfg.model.use_rope else None,
            attn_implementation="flash_attention_2" if cfg.model.use_flash_attention_2 else None,
            quantization_config=bnb_config if cfg.train.qlora.use else None,
        )
    )

    with Path.open(
        model_save_dir.joinpath(
            f"{cfg.eval.split}/{cfg.eval.domain if cfg.eval.domain else ''}/input_records.json",
        ),
        "r",
    ) as f:
        input_records = json.load(f)

    evaluate(cfg, model, tokenizer, input_records, result_dir)


def evaluate(
    cfg,
    model,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_records: list[dict[str, Any]],
    result_dir: Path,
) -> None:
    key_dataset = KeyDataset(input_records, key="text")  # type: ignore  # noqa: PGH003
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )

    results = []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore  # noqa: PGH003
        pbar = tqdm(
            pipe(
                key_dataset,
                batch_size=cfg.eval.batch_size_per_device,
                pad_token_id=tokenizer.unk_token_id,
                max_new_tokens=cfg.model.max_out_len,
                return_full_text=False,
            ),
            desc="Generating outputs",
            total=len(key_dataset),
        )
        for i, out in enumerate(pbar):
            input_record = input_records[i]
            generated_text = out[0]["generated_text"]
            result = {
                "demo_name": input_record["demo_name"],
                "turn_index": input_record["turn_index"],
                "prompt": input_record["prompt"],
                "text": input_record["text"],
                "output_predicted": generated_text,
                "output_target": input_record["output_target"],
                "output_target_dict": input_record["output_target_dict"],
            }

            results.append(result)

    with Path.open(result_dir.joinpath("results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

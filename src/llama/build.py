import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import huggingface_hub
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from weblinx import Demonstration
from weblinx.processing import load_candidate_elements
from weblinx.processing.prompt import (
    build_input_records_from_selected_turns,
    select_turns_and_candidates_for_prompts,
)
from weblinx.utils import (
    load_demo_names_in_split,
    set_seed,
)

from .processing import (
    build_formatter_for_multichoice,
    build_prompt_records_for_llama_truncated,
    insert_formatted_chat_into_records,
)

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
    input_records_save_dir = Path(model_save_dir.joinpath(f"{cfg.build.split}")).expanduser()
    input_records_save_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_name,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    split_path = Path(cfg.data.split_path).expanduser()

    demo_names: list[str] = load_demo_names_in_split(split_path, split=cfg.build.split)
    demos = [Demonstration(demo_name, base_dir=cfg.data.base_dir) for demo_name in demo_names]
    candidates = load_candidate_elements(path=cfg.candidates.build_path)

    format_intent = build_formatter_for_multichoice()
    build_prompt_records_fn = partial(
        build_prompt_records_for_llama_truncated,
        format_intent=format_intent,
        tokenizer=tokenizer,  # type: ignore  # noqa: PGH003
    )

    selected_turns: list[dict[str, Any]] = select_turns_and_candidates_for_prompts(
        demos=demos,
        candidates=candidates,
        num_candidates=cfg.candidates.k,
    )

    input_records = build_input_records_from_selected_turns(
        selected_turns=selected_turns,
        format_intent=format_intent,
        build_prompt_records_fn=build_prompt_records_fn,
        format_prompt_records_fn=None,
    )

    input_records = insert_formatted_chat_into_records(
        input_records,
        demos,
        tokenizer,
        include_output_target=cfg.build.include_output_target,
    )

    with Path.open(input_records_save_dir.joinpath("input_records.json"), "w") as f:
        json.dump(input_records, f, indent=2)


if __name__ == "__main__":
    main()

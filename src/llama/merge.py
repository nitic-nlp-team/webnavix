import logging
from pathlib import Path

import huggingface_hub
import hydra
import torch
from mergoo.compose_experts import ComposeExperts
from omegaconf import OmegaConf
from weblinx.utils import set_seed


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    huggingface_hub.login(token=cfg.huggingface.token)

    set_seed(cfg.seed)

    model_save_dir = Path(cfg.model.save_dir).expanduser()
    model_save_dir.mkdir(exist_ok=True, parents=True)

    merge_config = {
        "model_type": "llama",
        "num_experts_per_tok": OmegaConf.to_container(cfg.merge.num_experts_per_tok, resolve=True),  # type: ignore  # noqa: PGH003
        "experts": OmegaConf.to_container(cfg.merge.experts, resolve=True),  # type: ignore  # noqa: PGH003
        "router_layers": OmegaConf.to_container(cfg.merge.router_layers, resolve=True),  # type: ignore  # noqa: PGH003
    }
    merger = ComposeExperts(merge_config, torch_dtype=torch.bfloat16)
    merger.compose()
    merger.save_checkpoint(model_save_dir)


if __name__ == "__main__":
    main()

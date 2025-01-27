import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    elm_utilization_rate_data = np.zeros((32, 8))
    for i in range(32):
        file_path = f"{cfg.analysis.data_path}/{i}.csv"
        data = pd.read_csv(file_path, header=None)

        result = {}

        for _, row in data.iterrows():
            domain = int(row.iloc[-1])
            array = row.iloc[:-1].to_list()
            if domain not in result:
                result[domain] = []
            result[domain].append(array)

        avg = []

        for domain in sorted(result.keys()):
            arrays = np.array(result[domain])
            mean_vector = np.mean(arrays, axis=0)
            avg.append(mean_vector)

        avg = np.array(avg).T

        elm_utilization_rate_data[i] = np.mean(avg, axis=1)

    plt.figure(num=file_path, figsize=(8, 6))

    fig, ax = plt.subplots()
    im = ax.imshow(elm_utilization_rate_data.T, cmap="viridis", origin="lower")

    ax.set_xlabel("Domain")
    ax.set_ylabel("ELM")

    plt.colorbar(im)

    plt.savefig(f"{cfg.analysis.data_path}/image/elm-utilization-rate-data.svg", format="svg", transparent=True)
    plt.close(fig)

    plt.figure(num=file_path, figsize=(8, 6))

    fig, ax = plt.subplots()
    im = ax.imshow(elm_utilization_rate_data[1:-1].T, cmap="viridis", origin="lower")

    num_domains = elm_utilization_rate_data.shape[0] - 2
    xticks_positions = np.arange(0, num_domains, 5)
    xticks_labels = xticks_positions + 1
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)

    ax.set_xlabel("Domain")
    ax.set_ylabel("ELM")

    plt.colorbar(im)

    plt.savefig(f"{cfg.analysis.data_path}/image/elm-utilization-rate-data-hidden.svg", format="svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

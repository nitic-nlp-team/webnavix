# WebNavix: Continuous Generalist Web Navigation Agent using Domain-wise Mixture-of-Experts

WebNavix is a continuous generalist web navigation agent that merges individually fine-tuned LLMs as domain experts.

## Core Contributors 🛠️

|                                           shio                                           |                                           ituki                                            |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| [<img src="https://github.com/dino3616.png" width="160px">](https://github.com/dino3616) | [<img src="https://github.com/ituki0426.png" width="160px">](https://github.com/ituki0426) |
|                   `#repository-owner` `#main-author` `#model-composer`                   |                               `#co-author` `#model-analyst`                                |

## Setup with Dev Containers 📦

You can easily launch the development environment of WebNavix with Dev Containers.  
Here is the step-by-step guide.

### Attention

- You need to install [Docker](https://docs.docker.com/get-docker) and [VSCode](https://code.visualstudio.com) before.

### 1. clone git repository

```bash
git clone "https://github.com/nitic-nlp-team/webnavix" && cd "./webnavix"
```

### 2. set environment variables

See `.env.example` or contact the [repository owner](https://github.com/dino3616) for more details.

### 3. launch dev containers

Launch containers using the VSCode extension [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

### 4. pin python version

```bash
rye pin $(cat "./.python-version")
```

### 5. install dependencies

```bash
rye sync
```

### 6. activate virtual environment

```bash
source "./.venv/bin/activate"
```

### 7. install FlashAttention-2

```bash
uv pip install flash-attn --no-build-isolation
```

## Setup locally 🖥️

If you want to build an environment more quickly without Docker, you can follow these steps to build your environment locally.

### Attention

- You need to install [rye](https://rye.astral.sh/guide/installation) before.
- [Optional] You should install project recommended VSCode extensions that specified in [`.devcontainer/devcontainer.json`](./.devcontainer/devcontainer.json#L8C7-L17C8) before.

### 1. clone git repository

```bash
git clone "https://github.com/nitic-nlp-team/webnavix" && cd "./webnavix"
```

### 2. set environment variables

See `.env.example` or contact the [repository owner](https://github.com/dino3616) for more details.

### 3. pin python version

```bash
rye pin $(cat "./.python-version")
```

### 4. install dependencies

```bash
rye sync
```

### 5. activate virtual environment

```bash
source "./.venv/bin/activate"
```

### 6. install FlashAttention-2

```bash
uv pip install flash-attn --no-build-isolation
```

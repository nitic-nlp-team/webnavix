# WebNavix: Generalist Web Navigation Agent using Domain-wise Mixture-of-Experts

Generalist Web Navigation Agent using domain-wise Mixture-of-Experts that merges individually fine-tuned LLMs as domain experts.

## Setup

### clone git repository

```bash
git clone "https://github.com/nitic-nlp-team/webnavix/"
```

### launch conatiner

```bash
docker compose -f "./docker/docker-compose.development.yaml" -p "webnavix" up -d
```

### install FlashAttention-2

```bash
uv pip install flash-attn --no-build-isolation
```

### set env variables

See `.env.example` for more details.

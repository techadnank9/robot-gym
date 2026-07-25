# Docker Notes

This project intentionally does not pin a specific Isaac image tag because NVIDIA distribution paths and tags change. Supply a real compatible base image through:

```bash
export ISAAC_BASE_IMAGE="<official-compatible-isaac-lab-or-isaac-sim-image>"
```

Then build:

```bash
docker compose -f docker/docker-compose.yaml build
```

The container mounts:

- repository source
- `outputs/`
- Isaac cache directories

It also exposes:

- API port
- dashboard port
- livestream ports defined in `config/livestream.yaml`

If the selected image does not provide Isaac Sim or Isaac Lab runtime bits, `scripts/check_isaac.sh` and the main runner fail loudly.

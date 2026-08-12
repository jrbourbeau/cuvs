# Milvus GPU benchmark deployment

This starts the official Milvus 3.0.0 GPU standalone deployment on GPU 0:

```bash
docker compose -f deploy/milvus/docker-compose.yml up -d
```

Select another GPU with `GPU_ID=1`. Install the client dependency and run a
small end-to-end benchmark before starting a full parameter sweep:

```bash
python -m pip install -e 'python/cuvs_bench[milvus]'
python -m cuvs_bench.run \
  --dataset sift-128-euclidean \
  --dataset-path /path/to/datasets \
  --backend-config deploy/milvus/backend.yaml \
  --algorithms milvus_gpu_cagra \
  --groups test --build --search
```

Remove `--groups test` to run the default GPU_CAGRA sweep. Stop the deployment
with `docker compose -f deploy/milvus/docker-compose.yml down`; its data remains
under `deploy/milvus/volumes/`.

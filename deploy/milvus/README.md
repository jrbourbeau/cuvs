# Milvus GPU benchmark deployment

This starts the official Milvus 3.0.0 GPU standalone deployment on GPU 0:

```bash
docker compose -f deploy/milvus/docker-compose.yml up -d --wait
```

Select another GPU with `GPU_ID=1`. Install the client dependency and run a
small end-to-end benchmark before starting a full parameter sweep:

```bash
uv pip install \
  click matplotlib pandas pyyaml requests 'scikit-learn>=1.5' \
  'pymilvus>=3.0.1'
RAPIDS_DISABLE_CUDA=true uv pip install --no-deps -e python/cuvs_bench
python -m cuvs_bench.run \
  --dataset sift-128-euclidean \
  --dataset-path ${PWD}/datasets \
  --backend-config deploy/milvus/backend.yaml \
  --algorithms milvus_gpu_cagra \
  --count 10 --batch-size 10000 --search-mode latency \
  --groups test --build --search
```

`--no-deps` uses cuVS and the common benchmark dependencies from the active
development environment instead of resolving the unpublished `cuvs==26.10.*`
source version. `RAPIDS_DISABLE_CUDA=true` only prevents the editable package
name from being rewritten to `cuvs-bench-cu12` or `cuvs-bench-cu13`; neither
option disables GPU support in Milvus.

Before benchmarking, `docker compose -f deploy/milvus/docker-compose.yml ps`
should report the `milvus` service as healthy. If startup fails, inspect it with
`docker compose -f deploy/milvus/docker-compose.yml logs milvus`.

Remove `--groups test` to run the default GPU_CAGRA sweep. Stop the deployment
with `docker compose -f deploy/milvus/docker-compose.yml down`. Docker-managed
volumes preserve its data; add `--volumes` to remove them as well.

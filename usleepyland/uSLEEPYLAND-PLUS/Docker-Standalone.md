# Run the inference engine in standalone Docker

This optional workflow packages only `uSLEEPYLAND-PLUS/` and its `weights/` folder. It runs `run.py` as a batch command, without the SLEEPYLAND-PLUS web interface. Start with the [engine README](README.md) for supported models, recognized channels, preprocessing, and output formats. The Dockerfile below is a packaging example and has not been validated as a separate release image; test it with a representative EDF before using it for a batch.

## Prepare the engine

Use a Linux x86-64 host with Docker Engine, enough disk space for the image and model weights, and internet access for the first build. Extract or copy the complete `uSLEEPYLAND-PLUS/` directory, including `run.py`, `environment.yaml`, `requirements.txt`, `usleepy/`, and `weights/`.

From that directory, create a `Dockerfile.standalone`:

```dockerfile
FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/uSLEEPYLAND-PLUS
COPY environment.yaml requirements.txt ./
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -f environment.yaml \
    && conda run -n usleepyland-plus python -m pip install --no-cache-dir 'pip<23' \
    && conda run -n usleepyland-plus python -m pip install --no-cache-dir -r requirements.txt \
    && conda clean -afy \
    && rm -rf /root/.cache/pip /root/.cache/conda
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "usleepyland-plus", "python", "run.py"]
```

Build and check the CLI:

```bash
docker build --platform linux/amd64 -f Dockerfile.standalone -t usleepyland-plus:local .
docker run --rm usleepyland-plus:local --help
```

The image includes the bundled weights. It does not publish a network port. The example uses CPU inference; GPU setup requires a separately validated image and host configuration.

## Run one recording

Create host folders and put a representative EDF with recognized EEG and/or EOG channels in `input/`:

```bash
mkdir -p /srv/sleepyland-plus/input /srv/sleepyland-plus/output
```

Then run the image, mounting the EDF folder read-only and results folder writable:

```bash
docker run --rm \
  --mount type=bind,src=/srv/sleepyland-plus/input,dst=/data,readonly \
  --mount type=bind,src=/srv/sleepyland-plus/output,dst=/results \
  usleepyland-plus:local \
  --edf_file /data/recording.edf \
  --output_root /results \
  --site_id study-001 \
  --model_name usleep,deepresnet,sleeptransformer \
  --lead_types eeg,eog \
  --pred_sec 30 \
  --model_set adult \
  --run_ensemble \
  --ensemble_method soft_vote \
  --review_software somnomedics
```

The container exits after inference. The host receives results in `/srv/sleepyland-plus/output/study-001/`. For `recording.edf`, the combined result is under `ensemble/30s/_all/soft-voting/`; its `recording_PRED.npy` contains probabilities and `recording_PRED.annot` contains the selected stage and all five probabilities per 30-second row. The `somnomedics/` subfolder holds import files when export succeeds. Check `sleep_model_extraction.log` if a requested model, lead, or export is missing.

For a batch, replace `--edf_file /data/recording.edf` with `--edf_dir /data`. The engine scans recursively for `*.edf`. Use distinct EDF filename stems and a fresh `--site_id` for each batch to avoid mixing results.

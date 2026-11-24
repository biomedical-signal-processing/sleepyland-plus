FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/app/uSLEEPYLAND \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app/uSLEEPYLAND

# Install system libraries and Miniconda before copying application code.
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates bzip2 gcc g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
        amd64) conda_arch=x86_64 ;; \
        arm64) conda_arch=aarch64 ;; \
        *) echo "Unsupported architecture: $TARGETARCH"; exit 1 ;; \
       esac \
    && wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${conda_arch}.sh" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# Dependency layers change only when the environment specifications change.
COPY uSLEEPYLAND-PLUS/environment.yaml uSLEEPYLAND-PLUS/requirements.txt ./
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -f environment.yaml \
    && conda run -n usleepyland-plus python -m pip install --no-cache-dir 'pip<23' \
    && conda run -n usleepyland-plus python -m pip install --no-cache-dir \
        python-multipart fastapi==0.95.0 pydantic==1.10.2 \
        'typing_extensions>=4.0.0' uvicorn pyyaml httpx \
    && conda run -n usleepyland-plus python -m pip install --no-cache-dir -r requirements.txt \
    && conda clean -afy \
    && rm -rf /root/.cache/pip /root/.cache/conda

# Model weights and source are copied last, so edits do not reinstall packages.
COPY uSLEEPYLAND-PLUS/ ./
COPY src/ ./src/

# scikit-learn's bundled libgomp has a per-architecture hash in its name; link it to a fixed path.
RUN ln -s /opt/conda/envs/usleepyland-plus/lib/python3.9/site-packages/scikit_learn.libs/libgomp-*.so* /opt/libgomp-sklearn.so \
    && test -e /opt/libgomp-sklearn.so
ENV LD_PRELOAD=/opt/libgomp-sklearn.so

EXPOSE 7777
CMD ["conda", "run", "--no-capture-output", "-n", "usleepyland-plus", "uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "7777"]

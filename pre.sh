apt-get update \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-get install -y software-properties-common \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    && apt-get update \
    && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    dh-autoreconf \
    libcudnn8-dev=8.2.0.53-1+cuda11.3 \
    && rm -rf /var/lib/apt/lists/*

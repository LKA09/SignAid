FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYOPENGL_PLATFORM=egl
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/third_party/SignAvatars-main
COPY third_party/SignAvatars-main/requirements.txt /tmp/signavatars-requirements.txt
# Legacy pins may require a project-specific CUDA/PyTorch wheel. Install the
# compatible torch build first, then the repository requirements.
RUN python3.10 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    python3.10 -m pip install --no-cache-dir -r /tmp/signavatars-requirements.txt
ENTRYPOINT ["python3.10", "vis.py"]

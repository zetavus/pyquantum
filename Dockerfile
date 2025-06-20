# PyQuantum Docker Image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

LABEL maintainer="PyQuantum Team"
LABEL description="PyTorch-native Quantum Computing Library"
LABEL version="0.2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY pyquantum/ ./pyquantum/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY docs/ ./docs/

# Install PyQuantum in development mode
RUN pip install -e .

# Create directories for notebooks and experiments
RUN mkdir -p /workspace/notebooks /workspace/experiments /workspace/results

# Set up Jupyter config
RUN pip install jupyter jupyterlab
RUN jupyter lab --generate-config

# Expose ports for Jupyter
EXPOSE 8888

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🌟 PyQuantum Docker Container 시작됨!"\n\
echo "📊 GPU 상태 확인..."\n\
python -c "import torch; print(f\"CUDA 사용 가능: {torch.cuda.is_available()}\"); print(f\"GPU 개수: {torch.cuda.device_count()}\")" || echo "GPU 확인 실패"\n\
echo "🧪 PyQuantum 설치 확인..."\n\
python -c "from pyquantum import test_installation; test_installation()" || echo "PyQuantum 확인 실패"\n\
echo "🚀 준비 완료! 사용 가능한 명령어:"\n\
echo "  - python examples/bell_state.py     # 벨 상태 실험"\n\
echo "  - python examples/xor_qnn.py        # XOR 양자신경망"\n\
echo "  - jupyter lab --ip=0.0.0.0 --allow-root --no-browser  # Jupyter Lab 시작"\n\
echo "  - python tests/test_basic.py        # 테스트 실행"\n\
exec "$@"' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# Default command
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]
# ベースイメージにはMicrosoftのDev Container用Pythonイメージ(3.8)を使用
FROM mcr.microsoft.com/devcontainers/python:3.8

# システムパッケージの更新とインストール
# audio処理に必要な ffmpeg と libsndfile1 (soundfileで必要) を含めています
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# setup.py に記載された主要な依存関係を事前にインストールしてキャッシュさせます
# これによりコンテナ作成時の待ち時間を短縮します
RUN pip install --no-cache-dir \
    "torch>=1.13.0" \
    "torchaudio>=0.13.0" \
    "torchvision>=0.14.0" \
    "numpy>=1.24.0" \
    "transformers>=4.29.0" \
    "librosa>=0.10.0" \
    "torchlibrosa>=0.0.9" \
    tqdm \
    gradio \
    pyyaml \
    einops \
    chardet \
    soundfile \
    scipy \
    pandas \
    progressbar \
    ftfy

# 作業ディレクトリの設定（オプション）
WORKDIR /workspaces/AudioLDM
#!/bin/bash
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
    "zai-org/glm-4-9b-chat"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen2.5-7B"
)

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "正在下载: $model"
    echo "=================================================="
    source /etc/network_turbo
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download $model \
        --local-dir ~/autodl-tmp/models/$(echo $model | tr '/' '-') \
        --local-dir-use-symlinks False
done

echo "全部下载完成！所有模型都在 ~/autodl-tmp/models/"
du -sh ~/autodl-tmp/models/*
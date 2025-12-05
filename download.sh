#!/bin/bash
# download_all_english_llms.sh
# 一键把上面所有顶级英文开源 LLM 下载到 ~/autodl-tmp/models

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2-7B-Instruct"
)

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "正在下载: $model"
    echo "=================================================="
    huggingface-cli download $model \
        --local-dir ~/autodl-tmp/models/$(echo $model | tr '/' '-') \
        --local-dir-use-symlinks False
done

echo "全部下载完成！所有模型都在 ~/autodl-tmp/models/"
du -sh ~/autodl-tmp/models/*
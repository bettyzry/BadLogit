import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/home/server/Documents/zry/LongBD/models', revision='master')
# Qwen/Qwen2.5-VL-7B-Instruct
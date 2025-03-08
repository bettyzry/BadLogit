import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1', cache_dir='/home/server/Documents/zry/selfllm/autodl', revision='master')
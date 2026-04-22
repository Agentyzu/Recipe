from modelscope import snapshot_download

# Automatically downloads the model to ~/.cache/modelscope/hub/qwen/Qwen-VL-Chat
# This utility ensures the weights are available locally for the inference script.
model_dir = snapshot_download('qwen/Qwen-VL-Chat')

print(f"Model has been downloaded to: {model_dir}")
MODEL="google/gemma-3-1b-it"

# vLLM model args (bf16, use chat template, 128k context)
MODEL_ARGS='model_name=google/gemma-3-1b-it,dtype=bfloat16,use_chat_template=true,max_model_length=131072,gpu_memory_utilization=0.9'

# Evaluate MATH-500 (0-shot). Output to a folder and also write per-item details.
lighteval vllm "$MODEL_ARGS" "lighteval|math500|0|0" \
  --output-dir runs/gemma3-4b-it-math500 \
  --save-details
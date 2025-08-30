export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL="google/gemma-3-4b-it"
GEN='generation_parameters={"max_new_tokens":2048,"temperature":0.0}'

lighteval vllm \
  "model_name=$MODEL,dtype=bfloat16,use_chat_template=true,max_model_length=131072,gpu_memory_utilization=0.90,$GEN" \
  "lighteval|gpqa|0|0" \
  --output-dir runs/gemma3-4b-it-gpqa --save-details

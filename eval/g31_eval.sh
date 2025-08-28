export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL="google/gemma-3-1b-it"
GEN='generation_parameters={"max_new_tokens":2048,"temperature":0.0}'

lighteval vllm \
  "model_name=$MODEL,dtype=bfloat16,use_chat_template=True,max_model_length=32768,gpu_memory_utilization=0.90,$GEN" \
  "lighteval|math_500|0|0" \
  --output-dir runs/gemma3-1b-it-math_500 --save-details
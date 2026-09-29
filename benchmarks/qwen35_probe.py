import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
m='Qwen/Qwen3.5-2B'
tok=AutoTokenizer.from_pretrained(m)
model=AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda').eval()
torch.cuda.synchronize()
x=tok('JSON:',return_tensors='pt').to('cuda')
with torch.inference_mode():
    y=model.model(input_ids=x['input_ids'],use_cache=True,return_dict=True)
torch.cuda.synchronize()
print(json.dumps({'class':type(model).__name__,'lm_head':list(model.lm_head.weight.shape),'dtype':str(model.lm_head.weight.dtype),'alloc':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'hidden':list(y.last_hidden_state.shape),'device':torch.cuda.get_device_name(0)}))

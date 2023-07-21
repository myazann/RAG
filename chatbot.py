from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from utils import gptq_modelnames

class Chatbot:

    def __init__(self, repo_id) -> None:

        self.repo_id = repo_id
        self.prompt_template = self.get_prompt_template()
        self.gen_params = self.get_gen_params()
        self.is_gptq = self.check_is_gptq()

        self.tokenizer = self.init_tokenizer()
        self.model = self.init_model()

    def get_prompt_template(self):
        pass

    def get_gen_params(self):
        pass

    def check_is_gptq(self):
        return True if "GPTQ" in self.repo_id else False
    
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
    
    def init_model(self):

        if self.is_gptq:
            return self.get_GPTQ_model()
        else:
            return self.get_model()

    def get_model(self):
        pass
    
    def get_GPTQ_model(self):
    
        return AutoGPTQForCausalLM.from_quantized(
                self.repo_id,
                model_basename=gptq_modelnames[self.repo_id],
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
                inject_fused_attention=False,
                quantize_config=None)

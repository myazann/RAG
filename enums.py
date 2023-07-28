from enum import Enum
 
class REPO_ID(str, Enum):

    MPT_7B = "mosaicml/mpt-7b-chat"
    FALCON_7B = "tiiuae/falcon-7b-instruct"
    GPT4ALL_13B_GPTQ = "TheBloke/GPT4All-13B-Snoozy-SuperHOT-8K-GPTQ"
    VICUNA_7B_GPTQ = "TheBloke/vicuna-7B-v1.3-GPTQ",
    VICUNA_13B_GPTQ = "TheBloke/Vicuna-13B-1-3-SuperHOT-8K-GPTQ" 
    VICUNA_33B_GPTQ = "TheBloke/Vicuna-33B-1-3-SuperHOT-8K-GPTQ"
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA2_7B_GPTQ = "TheBloke/Llama-2-7b-Chat-GPTQ"
    LLAMA2_13B_GPTQ = "TheBloke/Llama-2-13B-chat-GPTQ"

class GPTQ_MODELNAMES(str, Enum):

    GPT4ALL_13B_GPTQ = "gpt4all-snoozy-13b-superhot-8k-GPTQ-4bit-128g.no-act.order"
    VICUNA_7B_GPTQ = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
    VICUNA_13B_GPTQ = "vicuna-13b-1.3.0-superhot-8k-GPTQ-4bit-128g.no-act.order"
    VICUNA_33B_GPTQ = "vicuna-33b-1.3-superhot-8k-GPTQ-4bit--1g.act.order"
    LLAMA2_7B_GPTQ = "gptq_model-4bit-128g"
    LLAMA2_13B_GPTQ = "gptq_model-4bit-128g"
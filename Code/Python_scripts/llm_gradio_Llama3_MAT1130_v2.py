import gradio as gr
import torch

import transformers

from transformers import pipeline
llama3_hf_token = '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'
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token = llama3_hf_token)
call_count = 0

text_file = "/home/vp899/projects/llm_pilot/Data/MAT1130_Context_OAI.txt"
# read the contents of text_file into str
with open(text_file, 'r') as file:
    input_text = file.read()

system_prompt = "You are a helpful digital assistant.\
    You will provide clear and concise answers. Your main reference is the text included within triple quotes. \
        The text is about MAT1130 which is the CNH Engineering Standard for Hydraulic Tubes''' " +  input_text + " '''"

def answer_function(message, chat_history):
    conversation = []
    n = 0
    conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.append({"role": "user", "content": user})
        conversation.append({"role": "assistant", "content": assistant[0]})
    conversation.append({"role": "user", "content": message})
    
    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)
    response = outputs[0][input_ids.shape[-1]:]

    llm_response = tokenizer.decode(response, skip_special_tokens=True)
    # return conversation1
    return llm_response


gr.ChatInterface(
    answer_function, 
    chatbot = gr.Chatbot(label = "My name is MAT1130", layout = "bubble", height = 700),
    textbox=gr.Textbox(placeholder="Ask me a question about MAT1130", container=False, scale=7),
    title="Ask me a question about MAT1130",
    theme="soft",
    cache_examples=True,
    retry_btn=None,
    submit_btn = None,
    undo_btn = None,
    fill_height=True,
    clear_btn = None).launch()


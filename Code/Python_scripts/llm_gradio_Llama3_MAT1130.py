import gradio as gr
import random

import transformers

from transformers import pipeline
llama3_hf_token = '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'
# pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token = llama3_hf_token, device=0)
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token = llama3_hf_token, device_map = "auto")
call_count = 0

text_file = "/home/vp899/projects/llm_pilot/Data/MAT1130_Context.txt"
# read the contents of text_file into str
with open(text_file, 'r') as file:
    input_text = file.read()
# print(text)
system_prompt = "You are a helpful digital assistant.\
    You will provide clear answers in 3 sentences. Your main reference is the text included within triple quotes.At the beginning of each answer,\
        you will include the main json tag from the portion of the reference text that you are using for the answer.''' " +  input_text + " '''"


def answer_function(message, chat_history):
    global call_count
    call_count += 1
    global system_prompt
    conversation = []
    # if len(chat_history) == 0:
      #  conversation.append({"role": "system", "content": system_prompt})
    # n = 0
    conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.append({"role": "user", "content": user})
        conversation.append({"role": "assistant", "content": assistant[0]})
    conversation.append({"role": "user", "content": message})
    print(pipe(conversation, max_new_tokens=128)[0]['generated_text'][-1])
    llm_response = pipe(conversation, max_new_tokens=500)[0]['generated_text'][-1]['content']
    # return conversation1
    return llm_response


gr.ChatInterface(answer_function, chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Ask me a question about MAT1130",
    description="Ask Yes Man any question",
    theme="soft",
    examples=["Hello", "Can you summarize MAT1130?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear").launch()


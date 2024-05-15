import gradio as gr
import random

import transformers

from transformers import pipeline
llama3_hf_token = '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'
# pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token = llama3_hf_token, device=0)
call_count = 0

def random_response(message, history):
    return random.choice(["Yes", "No"])

def answer_function(message, chat_history):
    global call_count
    call_count += 1
    conversation1 = ['System-1']
    conversation = []
    n = 0
    for user, assistant in chat_history:
    # for item in chat_history:
        print("User type: ",type(user),"----Assistant type: ", type(assistant), "Message: ",message, "---- call_count: ", call_count)
        print("User len: ",len(user),"----Assistant len: ", len(assistant), "Message: ",message, "---- call_count: ", call_count)
        print("User: ",user,"----Assistant: ", assistant, "Message: ",message, "---- call_count: ", call_count)
        # print("User: ",item[0],"----Assistant: ", assistant[0], "Message: ",message, "---- call_count: ", call_count)
        print("Conversation before: ", conversation)
        # conversation.append(user)
        conversation.append({"role": "user", "content": user})
        conversation.append({"role": "assistant", "content": assistant[0]})
        print("Conversation after: ", conversation)
        
        n += 1
        # concatenate n to message
        # message = message + "----" + str(n)
    # conversation.append({"role": "user", "content": message})
    # conversation.append(message)
    return conversation1


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


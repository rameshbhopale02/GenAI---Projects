import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("summarization", model="philschmid/distilbart-cnn-12-6-samsum", torch_dtype=torch.bfloat16)

model_path = ("../Models/models--philschmid--distilbart-cnn-12-6-samsum/snapshots/d8bbb0012bd874c792577caeaab58b6f72f64e2f")
text_summary = pipeline("summarization", model=model_path,
                torch_dtype=torch.bfloat16)



# text='''Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and investor.
# He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect,
# and former chairman of Tesla, Inc.; owner, executive chairman, and CTO of X Corp.;
# founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president
# of the Musk Foundation. He is one of the wealthiest people in the world; as of April 2024,
# Forbes estimates his net worth to be $178 billion.[4]'''
# print(text_summary(text))

def summary (input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="@GenAi : Text Summarizer",
                    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE TEXT")
demo.launch()
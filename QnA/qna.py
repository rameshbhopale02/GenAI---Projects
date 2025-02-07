import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

model_path = ("../Models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4")
question_answer = pipeline("question-answering", model=model_path,
                torch_dtype=torch.bfloat16)

# context = "Mark Elliot Zuckerberg (/ˈzʌkərbɜːrɡ/; born May 14, 1984) is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy."
# question = "what his date of birth?"


#
# answer = question_answer(context=context, question=question)
# print(answer["answer"])
def read_file_content(file_obj):
    """
    Reads the content of a file object and returns it.
    Parameters:
    file_obj (file object): The file object to read from.
    Returns:
    str: The content of the file.
    """
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as file:
            context = file.read()
            return context
    except Exception as e:
        return f"An error occurred: {e}"

def get_answer(file, question):
    context = read_file_content(file)
    answer = question_answer(context=context, question=question)
    return answer["answer"]

# print(get_answer())

gr.close_all()

demo = gr.Interface(
    fn=get_answer,
    inputs=[
        gr.File(label="Upload your file"),
        gr.Textbox(label="Query Question Based on Context")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="@GenAI : QnA with GenAi",
    description="THIS APPLICATION WILL BE USED TO ASK QUESTIONS BASED ON THE CONTEXT PROVIDED."
)
demo.launch()
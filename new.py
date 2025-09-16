import os
import warnings
import gradio as gr
from dotenv import load_dotenv;
from langchain.chains import LLMChain;
from langchain.chat_models import ChatOpenAI;
from langchain.prompts import PromptTemplate;

warnings.filterwarnings("ignore")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(model="gpt-3.5-turbo")

prompt_template=PromptTemplate(
    input_variables=["text"],
    template="""Please do the following:
1. Summarize the following text in a few sentences.
2. After the summary, create 10 multiple-choice quiz questions with answers based on the text.

Text: {text}

Answer in the following format:

Summary:
<your summary>

Quiz:
1. Question 1
   a) Option
   b) Option
   c) Option
   Answer: correct_answer
2. Question 2
...
    """
)

quiz_and_summary_chain=LLMChain(llm=llm,prompt=prompt_template)

def generate_quiz_and_summary(text):
    result=quiz_and_summary_chain.run(text)
    return result


iface=gr.Interface(
    fn=generate_quiz_and_summary,
    inputs=gr.Textbox(lines=10, label="Paste your text here"),
    outputs=gr.Textbox(label="Generated Summary and Quiz"),
    title="Text Summarizer and Quiz Generator",
    description="Paste your text and get a summary along with 10 quiz questions and answers"
)

iface.launch()
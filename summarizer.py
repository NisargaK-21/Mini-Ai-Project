#!/usr/bin/env python
# coding: utf-8

# In[1]:


import langchain


# In[3]:


langchain.__version__


# In[2]:


from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


# In[3]:


import os


# In[4]:


os.environ["OPENAI_API_KEY"] = "sk-"


# In[13]:


prompt_template=PromptTemplate(
    input_variables=["text"],
    template="""summarize the following text in a concise manner along with 10 quiz questions with answers at last :
    Text: {text}
    quiz:
    """
)


# In[14]:


llm=ChatOpenAI(model="gpt-3.5-turbo")


# In[15]:


import warnings
warnings.filterwarnings("ignore")  


# In[16]:


quiz_chain=LLMChain(llm=llm,prompt=prompt_template)


# In[17]:


example_text = """Text-based AI, also known as AI text generators or chatbots, are computer programs 
that generate human-like text based on prompts and large datasets. These tools, powered by Large Language Models (LLMs)
 and Natural Language Processing (NLP), can write essays, answer questions, create marketing copy, 
 and even generate computer code. Popular examples include ChatGPT, Gemini, 
and Claude, and they are widely used for content creation and increasing productivity across various fields. 
"""


# In[18]:


quiz = quiz_chain.run(text=example_text)
print(quiz)


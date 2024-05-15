import torch
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipelines
from dotenv import load_dotenv
import os



app = Flask(__name__)
CORS(app)
app.secret_key = 'medisation'

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(HF_TOKEN)

# Get the type
compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

from transformers import AutoModelForCausalLM, AutoTokenizer

hf_model_repo = "smrynrz20/llama-2-7b-qna-tuned"

# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)

# Load the model
model = AutoModelForCausalLM.from_pretrained(hf_model_repo,
                                             quantization_config=bnb_config,
                                             device_map="auto")

pipe = pipelines("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# tokenizer = AutoTokenizer.from_pretrained("smrynrz20/bart_qa_model")
# model = AutoModelForQuestionAnswering.from_pretrained("smrynrz20/bart_qa_model")


# llm=HuggingFaceHub(repo_id="smrynrz20/bart_qa_model",
#                     model_kwargs={"temperature":0,
#                                   "max_length":64})

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What is UTI?"

# print(llm_chain.invoke(question))

# chat_model = ChatHuggingFace(llm=llm, token=HF_TOKEN)
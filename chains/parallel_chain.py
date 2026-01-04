from dotenv import load_dotenv
import os

# LangChain Models using
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# Tokens
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------
# HuggingFace Zephyr Model
# ---------------------------
hf_llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=200,
    temperature=0.4,
    return_full_text=False,  # IMPORTANT: prevents [USER] leakage
)

model1 = ChatHuggingFace(llm=hf_llm)

# ---------------------------
# Gemini Model
# ---------------------------
model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# ---------------------------
# Prompts (CHAT SAFE)
# ---------------------------
notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher who writes short, clear study notes."),
    ("human", "Generate short and simple notes from the following text:\n{text}")
])

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an exam question generator."),
    ("human", "Generate exactly 5 short question-answer pairs from the following text:\n{text}")
])

merge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You combine notes and quiz into a clean study document."),
    ("human", """
Create a well-structured study document.

### Notes
{notes}

### Quiz (5 Q&A)
{quiz}
""")
])

# ---------------------------
# Output Parser
# ---------------------------
parser = StrOutputParser()

# ---------------------------
# Parallel Chain
# ---------------------------
parallel_chain = RunnableParallel({
    "notes": notes_prompt | model1 | parser,
    "quiz": quiz_prompt | model2 | parser
})

# ---------------------------
# Merge Chain
# ---------------------------
merge_chain = merge_prompt | model1 | parser

# ---------------------------
# Full Chain
# ---------------------------
chain = parallel_chain | merge_chain

# ---------------------------
# Input Text
# ---------------------------
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outliers detection.

They are effective in high dimensional spaces and are memory efficient because they use only a
subset of training points known as support vectors.

SVMs support linear and non-linear classification using kernel functions.
Common kernels include linear, polynomial, and RBF.

A disadvantage of SVMs is that choosing the correct kernel and regularization is crucial,
especially when the number of features exceeds the number of samples.
"""

# ---------------------------
# Run Chain
# ---------------------------
result = chain.invoke({"text": text})

print("\n================ FINAL OUTPUT ================\n")
print(result)

chain.get_graph().print_ascii()

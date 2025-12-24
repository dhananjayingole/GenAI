from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    max_new_tokens=200,
    temperature=0.4,
)

chat_model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.format()
# json format
result = chat_model.invoke(prompt)
print(result)
# structured form in dictionary of json.
final_result = parser.parse(result.content)
print(final_result)

# # through chain
# chain = template | chat_model | parser
# resulted = chain.invoke({})
# print(resulted)
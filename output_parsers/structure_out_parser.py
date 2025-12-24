from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

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

# Define schema
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt template
template = PromptTemplate(
    template="""
Give 3 facts about {topic}.
{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    },
)

# Chain
chain = template | chat_model | parser

# Invoke
result = chain.invoke({"topic": "black hole"})
print(result)

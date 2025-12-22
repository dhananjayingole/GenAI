import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import load_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize HuggingFace model
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check if token exists
if not HF_TOKEN:
    st.error("⚠️ HUGGINGFACEHUB_API_TOKEN not found in .env file!")
    st.info("Please add your Hugging Face token to the .env file")
    st.stop()

# Initialize the model
try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",  # Changed from conversational to text-generation
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    st.error(f"❌ Error initializing model: {str(e)}")
    st.stop()

# Streamlit UI
st.header('Research Tool')

# Paper selection
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Style selection
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# Length selection
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Load the prompt template
try:
    template = load_prompt('template.json')
except FileNotFoundError:
    st.error("⚠️ template.json file not found!")
    st.info("Please run prompt_generator.py first to create the template")
    st.stop()
except Exception as e:
    st.error(f"Error loading template: {str(e)}")
    st.stop()

# Summarize button
if st.button('Summarize'):
    with st.spinner('Generating summary...'):
        try:
            # Format the template with user inputs
            prompt_content = template.format(
                paper_input=paper_input,
                style_input=style_input,
                length_input=length_input
            )
            
            # Create message for the chat model
            messages = [HumanMessage(content=prompt_content)]
            
            # Invoke the model
            response = chat_model.invoke(messages)
            
            # Display the result
            st.write(response.content)
            
        except Exception as e:
            st.error(f"❌ Error generating summary: {str(e)}")
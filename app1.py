import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLama 2 model
def getLLamaFunc(input_text, no_words, blog_style):
    # Llama2 model calling
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    # Prompt template     
    template = """Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."""
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)

    # Generate the response from llama2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Load your dataset
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Streamlit app
def main():
    st.set_page_config(page_title="Generate Answer",
                       layout='centered',
                       initial_sidebar_state='collapsed')

    st.header("Generate Answers")

    # Load data
    data = load_data("code_data.csv")

    # Sidebar inputs
    input_text = st.text_input("Enter your query")
    col1, col2 = st.columns([5, 5])
    with col1:
        no_words = st.text_input("No of words")
    with col2:
        blog_style = st.selectbox("Writing the answer for", ('Users','Admin','Researcher', 'etc'),index=0)

    # Button to generate response
    submit = st.button("Generate")

    # Display response
    if submit:
        response = getLLamaFunc(input_text, no_words, blog_style)
        st.write(response)

if __name__ == "__main__":
    main()



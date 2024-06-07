import openai
import os
import streamlit as st
from llama_index import download_loader
from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_hub.file.unstructured.base import UnstructuredReader
# from transformers import pipeline
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import OpenAI
from datetime import datetime


openai.api_key = ""

# llm = LlamaCPP()

# classifier = pipeline("summarization")
# @st.cache(allow_output_mutation=True)
# def load_files():
#     files_dict = {}
#     data_dir = './data'
#     # Get list of all .txt files in current directory
#     txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
#     print("txt_files:", txt_files)
#     for file in txt_files:
#         file_path = os.path.join(data_dir, file)
#         try:
#             with open(file_path, 'rb') as f:
#                 text = f.read().decode('utf-8', 'ignore')
#         except UnicodeDecodeError as e:
#             print(f"Error decoding {file}: {e}")
#             text = ""
#         files_dict[file] = text
        
#     return files_dict

# def get_summary(documents):
#     summary_dict = {}
#     for doc in documents:
#         summary = classifier(documents[doc][:1000])
#         summary_dict[doc] = summary
#     return summary_dict

# @st.cache(allow_output_mutation=True)
@st.cache(allow_output_mutation=True)
def create_index():
    file_metadata = lambda x: {"filename": x}
    documents = SimpleDirectoryReader('./data', file_metadata=file_metadata).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return documents, index


# files_dict = load_files()
# st.write(files_dict)
    
documents, index = create_index()

# st.write("loaded docs:",documents)

@st.cache_data
def get_summary_llama2():
    summary_dict = {}
    idx = 0
    for doc in documents:
        # prompt = f"Extract the date and summarize the below text, return it as key value pair:\n{doc.text[:2000]}"
        # st.write(doc)
        prompt1 = f"""You job is to extract date in %Y-%m-%d format from the below text, it no date is found return 2000-01-01, only return the date:\n{doc.text[:3000]}"""
        response1 = OpenAI().complete(prompt1)
        prompt2 = f"""You job is to extract summarize below text including key points:\n{doc.text[:3000]}"""
        response2 = OpenAI().complete(prompt2)

        print(response1.text)
        summary_dict[doc.metadata['filename']] = {'date':response1.text.strip('.'), 'summary':response2.text}
        # summary_dict = dict(sorted(summary_dict.items(), key=lambda x: datetime.strptime(x[1]['date'], '%Y-%m-%d')))

        idx+=1
    # print(response.text)
    return summary_dict
summaries = get_summary_llama2()
st.write(summaries)
# st.write(documents)



query_engine = index.as_query_engine()
# response = query_engine.query("who is the author")
user_query = st.text_input("Enter your query", value="evidence of fraud")
if len(user_query) > 2:
    response = query_engine.query(user_query)
    st.write(response.response)
    st.write(response.metadata)
    # st.write(response)
# print(response)
# print(type(response))


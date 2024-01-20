import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template

from utils.pdf_utils import MyPDF
from utils.text_splitter_utils import MyTextSplitter
from utils.vector_store_utils import MyVectorStore
from utils.langchain_utils import MyLangChain
import json


def get_conversation_chain(retriever):
    my_lang_chain = MyLangChain()
    return my_lang_chain.generate_prompts_chain(base_retriever=retriever)


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error(f"Please enter document")
        return

    result = st.session_state.conversation.invoke(
        {
            "user_prompt": user_question,
            "num_of_prompts_to_generate": 5,
        }
    )
    prompts_generated = json.loads(result["response"].content)

    for message in prompts_generated:
        st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Optimized Prompts", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Get optimized prompts 🤖")
    user_question = st.text_input("State your objective:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                pdf = MyPDF(pdf=pdf_docs)
                raw_text = pdf.get_pdf_text()

                # get the text chunks
                text_splitter = MyTextSplitter(raw_text)
                text_chunks = text_splitter.get_text_chunks()

                # create vector store
                my_vector_store = MyVectorStore()
                chroma_vector_store = my_vector_store.embed_text_and_return_vectorstore(
                    text_chunks
                )
                retreiver = my_vector_store.get_retriever(chroma_vector_store)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(retreiver)


if __name__ == "__main__":
    main()

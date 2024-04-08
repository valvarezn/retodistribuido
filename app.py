from PyPDF2 import PdfReader
import os

## librerias lagchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

## librerias Streamlit 
import streamlit as st
## librerias para el response container
from streamlit_chat import message

#librerias text_container
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


## config st 
st.set_page_config(page_title= "ChatBot with PDF",layout="wide")
st.markdown("""<style>.block-container {padding-top:1rem;}</style>""", unsafe_allow_html=True)



#OPENIA_KEY
OPENAI_API_KEY="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"
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY


##creando llaves para la seccion_state 
session_state ={
    "responses":[],
    "requests":[]
}
if 'responses' not in st.session_state:
    st.session_state['responses']=[" Hola , ¿En que puedo ayudarte?"]

if 'requests' not in st.session_state:
    st.session_state['requests']=[]


##funcion crear caracterisitcas

def create_embeddings(pdf):
    #extraer PDF
    if pdf is not  None:
        pdf_reader= PdfReader(pdf)
        text =""
        for page in pdf_reader.pages:
            text +=page.extract_text()

        ## dividir en trozos 
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200, #super posicion entre oracion 
            length_function =len
        )

        chunks=text_splitter.split_text(text)

        embeddings= OpenAIEmbeddings()

        embeddings_pdf= FAISS.from_texts(chunks,embeddings)

        return embeddings_pdf
    
    # cargar documento 
st.sidebar.markdown("<h1 style='text-align:center; color:#176887;'> Cargar Archivo PDF </h1>",unsafe_allow_html=True)
st.sidebar.write("Cargar archivo .PDF con el cual desea chatear")

pdf_doc= st.sidebar.file_uploader("", type="pdf")

st.sidebar.write("---")

# crear embeddings
embeddings_pdf=create_embeddings(pdf_doc)


#chat seccion 
st.markdown("<h2 style='text-align:center; color:#888a8a; text-decoration:underline;'><strong> ChatBox PDF </strong></h2>",unsafe_allow_html=True)

# Verificar si se ha cargado un archivo PDF
if pdf_doc is not None:
    embeddings_pdf = create_embeddings(pdf_doc)
    st.success("¡Documento PDF cargado y embeddings creados con éxito!")
else:
    st.warning("Por favor, carga un archivo PDF en el sidebar para continuar.")
st.write("----")

# container del historial de chat
response_container = st.container()

# container de texto box
textcontaner = st.container()


## campo de ingreso de preguntas del usuario 
with textcontaner:
    #formulario tex input
    with st.form(key='my_form', clear_on_submit=True):
        query= st.text_area("Tu:", key='input', height=100)
        submit_button= st.form_submit_button(label='Enviar')

    if query:
        with st.spinner("escribiendo...."):
            #cosine simular con api Emmbeddings
            docs= embeddings_pdf.similarity_search(query)

            # 4 respuestas posibles 

            llm =OpenAI(model_name = "babbage-002")
            chain= load_qa_chain(llm,chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs ,question=query)
                print(cb) 

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)       


#configurando el campo para ver historial del chat 
with response_container:
    if st.session_state['responses']:

        for i in range (len(st.session_state['responses'])):
            #respuesta del bot
            message(st.session_state['responses'][i],key=str(i),avatar_style='pixel-art')
            
            # pregunta del usuario
            if i<len(st.session_state['requests']):
                message(st.session_state["requests"][i],is_user=True, key=str(i)+ '_user')


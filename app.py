###########################################################################
#        Máster en Ingeniería Informática - Universidad de Valladolid     #
#                                                                         #
#                           Trabajo Fin de Máster                         #
#                                                                         #
#   Implementación de técnicas de RAG (Retrieval Augmented Generation)    #
#   sobre LLM (Large Language Models) para la extracción y generación     #
#                de documentos en las Entidades Públicas                  #
#                                                                         #
#                 Realizado por Miguel Ángel Collado Alonso               #
#                                                                         #
###########################################################################

# Import necesarios para el proyecto

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from chainlit.input_widget import Select

import time

import PyPDF2

# Variable que actúa a modo de switch para activar o no el modo RAG
RAG = False 


# Función que se ejecuta al principio de abrir el chat

@cl.on_chat_start
async def on_chat_start():
    
    global RAG
    
    # Pregunta al usuario si quiere usar RAG o no

    res = await cl.AskActionMessage(
        content="""Hola, soy el asistente virtual de TFM-RAG.         
           ¿Deseas utilizar RAG?""",
        actions=[
            cl.Action(name="si", value="Si", label="✅ Si"),
            cl.Action(name="no", value="No", label="❌ No"),
        ],
    ).send()

    if res and res.get("value") == "Si":
        RAG = True
    else:
        RAG = False
        await cl.Message(
            content="Por favor, selecciona un MODELO en las opciones de abajo y realiza tus cuestiones! Nota: El modelo por defecto es LLAMA3.",
        ).send()

    # Define la configuración del modelo LLM, permite seleccionar entre los modelos Llama3, Gemma2 
    # o Phi3. Por defecto está seleccionado Llama3.

    settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="LLM - Modelo",
                    values=["llama3","llama3:70b","gemma2","phi3"],
                    initial_index=0,
                    )
                ]
            ).send()
    selected_model = settings["Model"]

    # Código que se ejecuta si el modo RAG está activado

    if RAG:

        # Le pide al usuario que suba un fichero para hacer RAG

        files = None 
        
        # Espera que el usuario suba el fichero

        while files is None:
            files = await cl.AskFileMessage(
                content="Por favor, sube un fichero PDF para comenzar la conversación",
                accept=["application/pdf"],
                max_size_mb=60,  # Tamaño máximo de fichero
                timeout=180,  # Define el tiemout                
                type="assistant_message",
            ).send()
       
    
        # La variable file almacena el fichero subido por el usuario
        file = files[0]

        # Inicio del temporizador
        start_time = time.time()

        # Configura las imágenes que se visualizan en el chat mientras espera o cuando está listo
        elements1 = [
        cl.Image(name="image", display="inline", path="wait.png",size="small")
        ]
        elements2 = [
        cl.Image(name="image", display="inline", path="ready.png", size="small")
        ]
        
        # Informa al usuario de que se está procesando el fichero
        msg = cl.Message(content=f"Procesando `{file.name}`...",type="user_message", elements=elements1)
        await msg.send()

                
        # Lectura y extración del texto del fichero
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Particionado del fichero en chunk de tamaño 1024
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)

        # Creación de metadatos para cada pedazo de fichero o chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Creación de los Embeddings y base de datos de vectores en Chroma
        
        # Selección de modelo de embeddings con el que se desea trabajar        
        #embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large") 
        #embeddings = OllamaEmbeddings(model="snowflake-arctic-embed")
                
        # Creación de la base de datos con el texto, los embeddings y los metadatos
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )

        # Inicializa el historial de mensajes de la conversación
        message_history = ChatMessageHistory()

        # Buffer de memoria para almacenar el contexto de la conversación
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Crea el chat con el modelo seleccionado, el documento para hacer RAG y el buffer de memroia
        chain = ConversationalRetrievalChain.from_llm(
            ChatOllama(
                model=selected_model, # selección de modelo
                temperature=0.4, # parámetro temperatura para ajustar la creatividad
                system="You are an expert research assistant who will analyze, correlate, and extract relevant information from the given context and answer questions asked by the user. Your output should be precise and accurate with source information. You will always speak in the Spanish language."            
            ),
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

        # Fin del temporizador
        end_time = time.time()

        # Calcular el tiempo transcurrido
        elapsed_time = end_time - start_time

    
        # Informa al usuario de que el sistema está listo
        msg.content = f"Procesado `{file.name}` realizado en `{elapsed_time:.2f}` segundos. Ahora, por favor selecciona un MODELO en las opciones de abajo y realiza tus cuestiones! Nota: El modelo por defecto es LLAMA3."
        msg.elements = elements2
        await msg.update()

        # Almacena el chat en la sesión de usuario
        cl.user_session.set("chain", chain)
        
    else:            
    
    # Codigo que se ejecuta si el modo RAG está desactivado

        # Selección de modelo LLM
        model = Ollama(model=selected_model,
                       temperature=0.4) # parámetro temperatura para ajustar la creatividad
        
        #Crea el prompt indicándole al modelo cómo debe comportarse
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a very knowledgeable robot that provides precise and eloquent answers to any type of question. Your output should be precise and accurate with source information. You will always speak in the Spanish language.",
                ),
                ("human", "{question}"),
            ]
        )
        
        # Ejecuta el chat
        runnable = prompt | model | StrOutputParser()

        # Almacena el chat en la sesión de usuario
        cl.user_session.set("runnable", runnable)


# Función que se ejecuta cuando se recibe un mensaje del usuario

@cl.on_message
async def main(message: cl.Message):

    global RAG

    # Codigo que se ejecuta si el modo RAG está activado

    if RAG:

        # Recupera el chat de la sesión de usuario
        chain = cl.user_session.get("chain")
        
        # Inicializa el controlador de devolución de llamada
        cb = cl.LangchainCallbackHandler()
        

        # Llama al chat con el contenido del mensaje del usuario, añadiendo que responda en español
        res = await chain.ainvoke(message.content+". Responde en español, incluso si no encuentras respuesta.", callbacks=[cb])    
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []  # Variable para almacenar el texto de las Fuentes del documento

        # Procado del documento para referenciar las fuentes
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"fuente_{source_idx+1}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            # Añade las fuentes a la respuesta del chat
            if source_names:
                answer += f"\nFuentes:  {', '.join(source_names)}"
            else:
                answer += "\nNo se ha encontrado fuentes"
        
        # Devuelve los resultados
        await cl.Message(content=answer, elements=text_elements).send()
    
    else:

        # Codigo que se ejecuta si el modo RAG está desactivado

        # Recupera el chat de la sesión de usuario
        runnable = cl.user_session.get("runnable") 

        # Inicializa el mensaje
        msg = cl.Message(content="")

        # Llama al chat con el contenido del mensaje del usuario, añadiendo que responda en español
        async for chunk in runnable.astream(
            {"question": message.content+". Responde en español, incluso si no encuentras respuesta."},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        
        # Devuelve los resultados
        await msg.send()

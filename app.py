
import os
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import chainlit as cl

llm_local = ChatOllama(model="mistral:instruct")
llm_groq = ChatGroq(model_name='mixtral-8x7b-32768')

# Depending on which model you want to use, you need to install Ollama. then you need to use the CLI to pull the model that you need to use.

@cl.on_chat_start
async def on_chat_start():
    file_path = 'datasets/CRPD.pdf'  # PDF path
    chroma_db_dir = "./chroma_db"  # Chroma vector store directory

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing the Convention on the Rights of Persons with Disabilities and Optional Protocol")
    await msg.send()

    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Initialize docsearch as None
    docsearch = None

    # Check if the Chroma vector store already exists and is not empty
    if os.path.exists(chroma_db_dir) and os.listdir(chroma_db_dir):
        try:
            # Load existing Chroma vector store
            print("Loading existing Chroma Vector Store")
            docsearch = CommunityChroma(persist_directory=chroma_db_dir, embedding_function=embeddings)
            print("Chroma Vector Store loaded")
        except Exception as e:
            print(f"Error loading Chroma Vector Store: {e}")
            docsearch = None

    if docsearch is None:
        print("Creating a new Chroma Vector Store")
        # Read the PDF file
        pdf = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)

        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Create a Chroma vector store
        docsearch = await cl.make_async(CommunityChroma.from_texts)(
            texts, embeddings, metadatas=metadatas, persist_directory=chroma_db_dir
        )

        print("Chroma Vector Store created and saved")

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file_path}` done. You can now ask questions!"
    await msg.update()
    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    # Call backs happens asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    # Return results
    await cl.Message(content=answer).send()

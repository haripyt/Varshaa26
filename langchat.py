import os
import pickle
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from Crypto.Cipher import AES
from groq import Groq

################### Key Decryption #######################

def Decrypt(key, ciphertext, messageDigest, nonce):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)

    try:
        cipher.verify(messageDigest)  # verify if the original hash matches the output
        return plaintext.decode('utf-8')
    except ValueError:
        print("Key incorrect or message corrupted")



key = b'09865rfqghlafgtz78nafg3q'  # 24 bytes key        

################# Load Encrypted Data ####################

encrypted_filename = "./encrypted_data.pkl"

with open(encrypted_filename, "rb") as pickle_file:
    loaded_encrypted_data = pickle.load(pickle_file)

#################### SETUP API KEY ENVIRONMENT ####################

c_text = loaded_encrypted_data["ciphertext"]    
md = loaded_encrypted_data["message_digest"]
nonce = loaded_encrypted_data["nonce"]

def get_api_key():
    API_KEY = Decrypt(key, c_text, md, nonce)
    return API_KEY

API_KEY = get_api_key()


#################### Embedding and FAISS Setup ####################
model = SentenceTransformer('./Sentence_Transformer/all-MiniLM-L6-v2')

def doc_loader(doc_file):
    loader = PyPDFLoader(doc_file)
    pages_content = loader.load_and_split()
    return pages_content

def save_db(doc_file):
    contents = doc_loader(doc_file)
    content_texts = [doc.page_content for doc in contents]  # Extracting the actual content
    embeddings_val = model.encode(content_texts)
    embeddings_val = np.array(embeddings_val).astype('float32')
    
    # Initialize FAISS index for cosine similarity
    index = faiss.IndexFlatIP(embeddings_val.shape[1])
    faiss.normalize_L2(embeddings_val)
    index.add(embeddings_val)

    os.makedirs('faiss_embedding',exist_ok=True)

    # Save the FAISS index and the associated documents
    faiss.write_index(index, "./faiss_embedding/embedding_index.faiss")

    with open("./faiss_embedding/document_texts.pkl", "wb") as f:
        pickle.dump(content_texts, f)


###################### Groq API ##############################

def groq_api_response(user_query,most_relevant_docs):
    prompt = {
            'role': 'user',
            'content': f'''
            Instruction:1.Give correct answer from Context for Question, 
                      2.dont use html elements                         
            Question:{user_query},
            Context:{most_relevant_docs},
            Answer:
            ##Overview
            ##Treatment
            ##Advice
        '''
        }
    os.environ["GROQ_API_KEY"] = API_KEY    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

    chat_completion = client.chat.completions.create(
            messages=[prompt],
            model="llama-3.1-8b-instant",
        )

    res=chat_completion.choices[0].message.content

    return res

##################### Querying with FAISS #####################

def asking_query(user_query):
    # Load the FAISS index
    index = faiss.read_index("./faiss_embedding/embedding_index.faiss")
    
    # Load the original document contents (texts)
    with open("./faiss_embedding/document_texts.pkl", "rb") as f:
        content_texts = pickle.load(f)
    
    # Embed the user's query
    user_embedding = model.encode([user_query]).astype('float32')
    faiss.normalize_L2(user_embedding)

    D, I = index.search(user_embedding, k=2)  
    
    # Get the 3 most relevant documents based on the index
    most_relevant_docs = [content_texts[i] for i in I[0]]
    
    api_response=groq_api_response(user_query,most_relevant_docs)

    return api_response  # Return the top 3 most relevant documents

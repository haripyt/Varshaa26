import chainlit as cl
import os
import shutil  # Import the shutil module for removing folders
#from tempfile import NamedTemporaryFile
from langchat import save_db

import asyncio
from chatvqa import handle_message


global files_dirs

def dirs():
    files_dirs=os.listdir()
    return files_dirs

files_dirs=dirs()

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Database",
            markdown_description="**DB Create & Remove**",
            icon="https://static.vecteezy.com/system/resources/thumbnails/004/657/673/small/database-line-style-icon-free-vector.jpg",
        ),
        cl.ChatProfile(
            name="MED-BOT",
            markdown_description="**Med Bot LLAMA**",
            icon="https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/609/628/datas/original.png",
        ),
    ]


@cl.action_callback("Remove_DB")
async def on_action(action):
    folder_path='/content/faiss_embedding'
    shutil.rmtree(folder_path)
    await cl.Message(content=f" {action.name} executed {folder_path} removed!").send()


@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "Database":
        files_dirs=dirs()

        if 'faiss_embedding' in files_dirs:
            if 'embedding_index.faiss' and 'document_texts.pkl' in os.listdir('faiss_embedding'):
                print('db exist true')
                # Sending an action button within a chatbot message
                actions = [
                    cl.Action(name="Remove_DB", payload={"value":"example_value"}, label="Click")
                ]
                
                await cl.Message(content="Click 'Remove DB' Button to Delete Existing DB!", actions=actions).send()
                
        else:
            
            files = await cl.AskFileMessage(
                content="upload pdf file",accept=["application/pdf"],max_size_mb=20
            ).send()

            pdf_file = files[0]
            #print("pdf : #########",pdf_file)

            save_db(pdf_file.path)

            msg = cl.Message(content=f"Processed {pdf_file.name}, DB Files Created...")
            await msg.send()
    else:
        async def another_async_function():
          message=cl.Message(content="Hello User.")
          await cl.make_async(handle_message)(message)

        asyncio.run(another_async_function())

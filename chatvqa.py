import chainlit as cl
import cv2
from tensorflow.keras.models import load_model
from skimage.io import imread
import numpy as np
from skimage import transform

from langchat import asking_query
from langchat import get_api_key



import os
from groq import Groq

api_key = get_api_key()



async def predict_image(image_path):
    # Load the model
    model_path='./Models/Resnet152V2/med_dis_ResNet152V2_all_data.h5'
    model = load_model(model_path,compile=False)

    I_test=imread(image_path) 
    #plt.imshow(I_test)
    
    I_test=np.array(I_test).astype('float32')
    I_test=transform.resize(I_test, (224, 224, 3))
    I_test= np.expand_dims(I_test, axis=0)
    #I_test.shape

    predictions=model.predict(I_test)
    predictions_val=np.argmax(predictions)
    confidence=np.max(predictions)
    if confidence > 0.70 :
      classes = ["Benign Lung Cancer", "Malignant Lung Cancer", "Healthy Chest", "Healthy Lung","Phneumonia","Gliomas Brain Tumor","Meningioma Brain Tumor","Healthy Brain","Pituitary Brain Tumor"  ]
      prediction_class=classes[predictions_val]
      print(predictions)
    else:
      prediction_class="None"
      predictions_val="None"
    

    return prediction_class,predictions_val

@cl.on_message
async def handle_message(msg: cl.Message):
    pred=None
    sample_image='./green_tick.png'
    

    images = [file for file in msg.elements if "image" in file.mime]
    non_image_docs = [file for file in msg.elements if "image" not in file.mime and file.mime]
    
    final_value="Try again.."
    if images:
        image_path = images[0].path
        
        pred,predictions = await predict_image(image_path)

        if pred != "None":
            if pred in ["Healthy Chest", "Healthy Lung","Healthy Brain"]:
                query=f'with bold size headers and proper alignment , i want only about {pred} of overview and Tips.' 
            else:
                query=f'with bold size headers and proper alignment ,i want only about {pred} of Overview and Treatment List and advice.'
            res1=asking_query(query) 

            prediction_result=f'It has {pred}.'
            final_value = f"{prediction_result}\n\n{res1}"
        if pred == "None":
            final_value ="we cant provide any information about this image. try again with trained images."
            
            
    elif msg.content and not msg.elements:
        #res2=asking_query(msg.content)
        prompt = {
        'role': 'user',
        'content': f''' Instruction:
                        1.If Question is not relevent under ""Benign Lung Cancer", "Malignant Lung Cancer", "Healthy Chest", "Healthy Lung","Phneumonia","Gliomas Brain Tumor","Meningioma Brain Tumor","Healthy Brain","Pituitary Brain Tumor""
                            then print only "This is Out of Scope Question please try again with Lung Cancer, Brain Tumor, Phneumonia" otherwise respond answer to the question
                        2.Make sure Answer is correct with Question with proper headings
                        3.Dont mention about our prompt 
                        
            Question:{msg.content}

        '''
        }
        print("##############",api_key)
        # Use the API key in your code
        os.environ["GROQ_API_KEY"] = api_key
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        chat_completion = client.chat.completions.create(
            messages=[prompt],
            model="llama-3.1-8b-instant",
        )

        final_value=chat_completion.choices[0].message.content
            
    elif non_image_docs:
        final_value="Image File Only Support."

    
    await cl.Message(content=final_value).send()


    

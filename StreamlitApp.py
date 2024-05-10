import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging


#loading json files

with open ("D:\mcqgen1\Response.json","r") as file:
    RESPONSE_JSON=json.load(file)


#create a tile for the app
st.title("MCQs Creator Application with LangChain")


#create a form using st.form
with st.form("user_inputs"):
    #File Load
    uploaded_file=st.file_uploader("upload a pdf or txt")

    #input fields
    mcq_count=st.number_input("No. of mcq's", min_value=3, max_value=50)

    subject=st.text_input("insert subject",max_chars=20)

    #QuizTone
    tone=st.text_input("complexity level of questions",max_chars=20,placeholder='simple')

    #button
    button=st.form_submit_button("Create MCQ's")


    #check if button is clicked & all fileds have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading...."):
            try:
                text=read_file(uploaded_file)

                #count token & cost of API
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject":subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                        )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")

                if isinstance(response,dict):
                    #Extract quix data from the response
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)

                            #Display the review in text box as well

                            st.text_area(label='Review',value=response["review"])
                        else:
                            st.error("error in table data")

                else:
                    st.write(response)



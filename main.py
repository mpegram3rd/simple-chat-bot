from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
import boto3
import streamlit as st

# bedrock client - Credit: https://www.youtube.com/watch?v=E1-mUfpeRu0

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-v2:1"

# langchain
# python / javascript lib used to build complex ai applications
# Called "langchain" because you can chain together prompts and dbs you're going to reference or web scrapers, etc.
# recommend starting with that framework.

llm = BedrockLLM(
    model_id=modelID,
    client=bedrock_client,
    # settings you can send into the model
    # max_tokens can be moved up or down to change cost
    # temperature impacts the amount of "creativity" the model is allowed.
    model_kwargs={"max_tokens_to_sample": 200, "temperature": 0.9}
)


def my_chatbot(language, style, freeform_text):
    in_the_style_of = ""
    if style:
        in_the_style_of = f"in the style of {style}"

    # prompt template lets you structure a prompt in a cookie cutter way
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template=f"You are a chat bot answer {in_the_style_of}. You are responding in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = prompt | llm

    response = bedrock_chain.invoke({'language': language, 'freeform_text': freeform_text},
                                    config={'callbacks': [ConsoleCallbackHandler()]})  # logging hook

    return response


# streamlit super simple website
st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["English", "Spanish", "French", "Korean"])
style = st.sidebar.selectbox("Style",
                             ["", "Dad Joke", "Mark Twain", "Bugs Bunny", "Elmer Fudd", "Barack Obama", "Donald Trump",
                              "Joe Biden"])

if language:
    freeform_text = st.sidebar.text_area(label="What is your question?",
                                         max_chars=100)  # save money don't put in too much context

if freeform_text:
    response = my_chatbot(language, style, freeform_text)
    st.write(response)

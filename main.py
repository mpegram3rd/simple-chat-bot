from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import boto3
import os
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

llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    # settings to can send into the model
    # max_tokens can be moved up or down to change cost
    # temperature impacts the amount of "creativity" the model is allowed.
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)


def my_chatbot(language, freeform_text):
    # prompt template lets you structure a prompt in a cookie cutter way
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template=f"You are a chat bot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})

    return response

print(my_chatbot("english", "How many tablespoons of minced garlic is equivalent to 3 cloves?"))
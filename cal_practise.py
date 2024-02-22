!pip install langchain
!pip install langchain-openai
!pip install chromadb
!pip install pypdf

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(openai_api_key='123', organization="gluon-meson",
openai_api_base='https://{your_host}', model='gpt-3.5-turbo', temperature=0, streaming=False)

embeddings = OpenAIEmbeddings(openai_api_key='123', organization="gluon-meson",
openai_api_base='https://{your_host}', model='text-embedding-ada-002')

import requests
import json
from pydantic import BaseModel, Field

class SquareNumber(BaseModel):
    """Call this to get the square number"""
    base: float = Field(description="In an exponentiation operation, the base represents the number that is raised to a certain power.")
    exponent: float = Field(description="the exponent represents the power to which the base number is raised.")

def power(base:int, exponent:int) -> int:
    return base ** exponent

class MultiNumber(BaseModel):
    """Call this to get the multi number"""

def multi(left:int, right:int) -> int:
    return left * right    


class PlusNumber(BaseModel):
    """Call this to get the plus number"""

def plus(left:int, right:int) -> int:
    return left + right    

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import  StructuredTool
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math expert"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_tools=[StructuredTool.from_function(func=plus, name="PlusNumber", description="Call this to get the plus number"),
             StructuredTool.from_function(func=multi, name="MultiNumber", description="Call this to get the multi number"),
             StructuredTool.from_function(func=power, name="SquareNumber", description="Call this to get the square number")]

agent = create_openai_tools_agent(llm, agent_tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)

result = agent_executor.invoke({"input":"Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"})
print(result["output"])

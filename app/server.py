from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes


from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware



from langserve import add_routes


#LANGCHAIN--------------------------------------------------------------------------
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_cohere import ChatCohere
from langchain_elasticsearch import ElasticsearchChatMessageHistory
from langchain_openai import ChatOpenAI

from langchain_groq import ChatGroq

#------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------
import json
import re
from typing import AsyncGenerator
# import os
#------------------------------------------------------------------------------------



app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)




@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

#DB LINK------------------------------------------------------------------------------
db = SQLDatabase.from_uri("sqlite:///transactions.db", sample_rows_in_table_info=0)

def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)

#--------------------------------------------------------------------------------------


with open('api_key.json') as config_file:
    config = json.load(config_file)
#GROQ SETUP---------------------------------------------------------------------------------------------
# groq_api_key=config['groq_api_key']
# llm = ChatGroq(
#             groq_api_key=groq_api_key, 
#             model_name="llama3-8b-8192"
#     )
#-------------------------------------------------------------------------------------------------------

#COHERE SETUP--------------------------------------------------------------------------
cohere_api_key=config['cohere_api_key']
llm = ChatCohere(model="command-r-plus", cohere_api_key=cohere_api_key)
#--------------------------------------------------------------------------------------


# Function to remove markdown tags from the SQL query
def remove_markdown_tags(text):
    return re.sub(r'```.*?```', '', text, flags=re.DOTALL)

#NL TO SQL--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template_nl2sql = """
- Please do not give empty answers.
- It's okay to generate the SQL queries which return NULL when the conditions are not met, for example for an unexisting date yet in the db like 31/3/2025.

Based on the table schema below, write a SQL query that would answer the user's question:
    {schema_arg}
    The database contains information about transactions specifically money transfers
    Each line or observation represent transaction. A transaction is money transfer between two parties.
    There only one table in the database. 
    It contains these columns, don't use any other column name in the queries except these:
    Type: The type of transaction
    Montant: The amount of money transferred
    Timestamp: The  date and time the transaction was done
    Origine: The id of the one who transfered or sent money to the other party
    Destination: The id of the one who received money from the other party
    Responds with few word. Don't start your answer with ' Sure! Here is ....'. Give only the sql query, answer directly



    Like in these examples:
        question: "How many P2P transactions are there in the database?"
        SQL Query: "SELECT COUNT(Type) FROM lambtech2024ia_dataset WHERE Type='P2P';"
     
    Please ensure that the SQL code generated does not include triple backticks (\`\`\`) at the beginning or end and avoids including the word "sql" within the output.
    Never enclose the queries in backticks like this `SELECT MIN(Milliseconds) FROM tracks` for example.
    You should also be able to retrieve information schema of the sqlite database using PRAGMA queries or selecting from the SQLITE_SCHEMA.
    - You should be able to answer any question about fintech fraudulent activities and methods.

    Question: {question}
    SQL Query:"""
prompt_nl2sql = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
            MessagesPlaceholder(variable_name="history"),
            ("human", template_nl2sql),
        ]
    )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

sql_response = (
    RunnablePassthrough.assign(
        # ------------------------------------------------------------------------------------------------------------------------
                # history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
                # history=history,
        # ------------------------------------------------------------------------------------------------------------------------
        schema_arg=get_schema)
    | prompt_nl2sql
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#SQL TO NL --------------------------------------------------------------------------------------------------------------------------------------------
template_sql2nl = """
- You are Kangam, an fraud detection and SQL expert adept at answering to questions about the content of the database.
- You should be able to answer any question about fintech fraudulent activities and methods.
- When you are greeted also answer nicely by greeting back and offering your assistance or help to the user as the expert you are
- When you're asked a question in french, answer in french.
- When you're thanked, answer and offer your help again.
- Do not give answers to questions you were not asked about.
- For queries with NULL return value or with empty entries, answer with "No entries were found in the database!" or say that you didn't find anything in the database, say it in a natural way, do not do further operations.
- Do not affirm something is in the database when it is not, for example for a unexisting motif, plateau... or saying there are calls for a an unexisting plateau, motif...in the database
- Answer naturally, concisely and directly to questions you're asked. Only greet when you're greeted.
- You don't need to introduce yourself whenver you're asked a question, answer directly
Like in this example SQL query returning NULL results:
        question: "How many DJFHGGHE transactions are there in the database?"
        SQL Query: "SELECT COUNT(Type) FROM lambtech2024ia_dataset WHERE Type='DJFHGGHE';"
        SQL Response: "I couldn't find any transactions in the database for that tyoe!"
        
- Based on the table schema below, question, sql query and sql response, write a natural language response:
{schema_arg}
 The database contains information about transactions specifically money transfers
    Each line or observation represent transaction. A transaction is money transfer between two parties.
    There only one table in the database. 
    It contains these columns, don't use any other column name in the queries except these:
    Type: The type of transaction
    Montant: The amount of money transferred
    Timestamp: The date and time the transaction was done
    Origine: The id of the one who transfered or sent money to the other party
    Destination: The id of the one who received money from the other party
    Responds with few word. Don't start your answer with ' Sure! Here is ....'. Give only the sql query, answer directly


Please ensure that the SQL code generated does not include triple backticks (\`\`\`) at the beginning or end and avoids including the word "sql" within the output.
Never enclose the queries in backticks like this `SELECT MIN(Milliseconds) FROM tracks` for example.

Do not provide explanations, just simple  sentences to answer in a natural way, like humans do.
Question: {question}
SQL Query: {query}
SQL Response: {response}
"""
prompt_sql2nl = ChatPromptTemplate.from_template(template_sql2nl)
#-----------------------------------------------------------------------------------

full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema_arg=get_schema,
        response=lambda vars: run_query(remove_markdown_tags(vars["query"])),
    )
    | prompt_sql2nl
    | llm
    
    
    # | my_handler
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # print(store)
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# Edit this to add the chain you want to add
add_routes(app, 
           chain_with_history,
           path="/bot",)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
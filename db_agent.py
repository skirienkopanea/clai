import sqlite3
import inspect
import sys
from colorama import Fore, Back, Style, init
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, CSVLoader
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from io import StringIO
import json

def get_query(question):
    schema = ""
    with open("emaildb.sql", "r", errors="ignore") as file:
        # Reading from a file
            schema = file.read()

    model = "gpt-4o"
    system_prompt = f"""You are a natural language to SQL generator. Replay exclusively with the SQL code.
    Take the user prompt and generate the SQL statment based on the create statements for this schema:
    {schema}
"""
    user_prompt = f"""User Prompt: {question}
    Additional requirements:
    Any column from the schema that exceeds 255 characters must be encapsulated inside SUBSTR(column_name, 1, 255).
    Always use case insensitive search for LIKE clauses and %search term% except when the search term is enclosed by single or double quotes in the user prompt.
    """
    try:
        client = OpenAI()

        response = client.chat.completions.create(
        model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred with query: {question}",e)
    return result.replace("```sql","").replace("```","")

def execute_query(query: str) -> str:
    result = []

    # Connect to the database (or create it if it doesn't exist)
    conn = sqlite3.connect('build/email.db')

    # Create a cursor object
    cursor = conn.cursor()
    
    queries = query.strip().split(";")

    for query in queries:
        if len(query) > 6:
            query += ";"
            
            cursor.execute(query)

            # Fetch all results
            rows = cursor.fetchall()
            
            column_names = []

            if cursor.description is not None:
                for description in cursor.description:
                    if description is not None:
                        column_names.append(description[0])

            # Create DataFrame from rows and column names
            df = pd.DataFrame(rows, columns=column_names)

            dict = {
            "query": query,
            "columns": column_names,
            "rows": rows
            }

            # Convert the DataFrame to a Markdown table
            result.append(dict)

            # Commit the changes
            conn.commit()

            # Close the connection
        conn.close()
    return result

def sqlquery(q):
    query = get_query(q)
    return execute_query(query)

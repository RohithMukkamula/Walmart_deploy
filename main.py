
import streamlit as st
import pandas as pd
import regex as re
import json
import warnings
from google.oauth2 import service_account
import ast

from langchain.tools import Tool
from langchain_core.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
import numpy as np
warnings.filterwarnings(action='ignore')



st.set_page_config(
    page_title="SQL Query Generator",
    page_icon='tredence-squareLogo-1650044669923.webp',
)


def get_gcp_credentials():
    """Load GCP credentials from Streamlit secrets and create a credentials object."""
    st.write(st.secrets["gcp"]["credentials"])

    # # Check if the secret is already a dictionary (usual case)
    # if isinstance(st.secrets["gcp"]["credentials"], dict):
    #     creds_dict = st.secrets["gcp"]["credentials"]
    # else:
    #     # If the secret is a string, parse it into a dictionary
    #     creds_dict = json.loads(st.secrets["gcp"]["credentials"])
    creds_json = ast.literal_eval(st.secrets["gcp"]["credentials"][3:][:-3])
    credentials = service_account.Credentials.from_service_account_info(creds_json)
    return credentials

def get_bigquery_client():
    """Create a BigQuery client using credentials from Streamlit secrets."""
    credentials = get_gcp_credentials()
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    return client





model = VertexAIEmbeddings(
        model_name = 'textembedding-gecko@003',
        pproject="wmt-ca-customer-insights-dev",
        location="us-central1",
        credentials = get_gcp_credentials()

    )

df = pd.read_csv(r"C:\Users\vn57bij\Downloads\examples 2(in) (1).csv")
data = df.apply(lambda row: {"input": row['PROMPT'], "query": row['OUTPUT']}, axis=1).to_list()
examples = {item['input']: item['query'] for item in data}



with open('my_list.json', 'r') as file:
    embeddings = json.load(file)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
def search(query, embeddings, sentences, top_k=2):
    query_embedding = model.embed_documents([query])[0]
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return [(sentences[idx], similarities[idx]) for idx in top_k_indices]


 

@st.cache_resource
def load_bigquery(uri):
    db = SQLDatabase.from_uri(uri)
    return db

def convert_bq_iterator_to_dataframe(iterator):
    rows = [list(row) for row in iterator]
    if not rows:
        return None
    column_names = [field.name for field in iterator.schema]
    df = pd.DataFrame(rows, columns=column_names)
    return df

def extract_sql_query(document):
    pattern = r"```sql\s+(.*?)\s+```"
    matches = re.findall(pattern, document, re.DOTALL)
    return matches

@tool
def get_data_dictionary(table_name: str) -> str:
    """
    Returns the data dictionary for the provided table name.
    Parameters:

    table_name (str): The name of the table to get the data dictionary for.
    Returns:

    str: A string representation of the data dictionary DataFrame.
    """
    st.write(table_name)
    try:
        # Extract the base name and construct the data dictionary table name.
        pattern = r"^['\"](.*)['\"]$"
    
        # Search for the pattern in the input string
        match = re.match(pattern, table_name)
        
        # If the pattern matches (string is in quotes), return the matched group without quotes
        if match:
            name_ref =  match.group(1)  # Return the string without quotes
        else:
            name_ref = table_name  # Return the original string if not in quotes

        name = name_ref.split('_Data_Dict')[0] + '_Data_Dict'
        st.write(name)
        # Execute the query to get the data dictionary.
        client = get_bigquery_client()
        query_job = client.query(f'SELECT * FROM {name}')  # API request
        # Wait for the query to finish and get the results.
        rows = query_job.result()
        # Convert the query results to a pandas DataFrame.
        df = convert_bq_iterator_to_dataframe(rows)
        # Return the DataFrame as a string without the index.
        return df.to_string(index=False)
    except:
        return "Data Dictionary is not there. Proceed by assuming the meaning of variables."

tools = [
    Tool(
        name="get_data_dictionary",
        description="Returns the data dictionary for the provided table name.",
        func=get_data_dictionary
    )
]

CREDENTIALS = {
    "user1": "password1",
    "user2": "password2"
}

def authenticate(username, password):
    if username in CREDENTIALS and CREDENTIALS[username] == password:
        st.session_state["logged_in"] = True
        st.success("Login successful!")
    else:
        st.error("Invalid username or password.")
import pandas as pd



def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    placeholder = st.empty()

    if st.session_state["logged_in"] == False:
        with placeholder.container():
            st.title("Login Page")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.button("Login")
        if login_button:
            authenticate(username, password)
            placeholder.empty()

    if st.session_state["logged_in"] == True:
        db = load_bigquery("bigquery://wmt-ca-customer-insights-dev/datawhiz")
        st.title('Walmart Table Query Tool')

        question = st.text_input("Input: ", key="input")
        st.write('\n')
        st.markdown('##### Records to Return (default: 100)')
        no_of_records = st.slider('hour', 10, 10000, 100)
        submit = st.button("Ask the question")

        client = bigquery.Client()


        def generate():
            # Initialize the Vertex AI LLM
            llm = VertexAI(
                project="wmt-ca-customer-insights-dev",
                location="us-central1",
                model="gemini-1.5-flash",
                credentials = get_gcp_credentials()
            )

            
            results = search(question, embeddings, list(examples.keys()))
            f_ex = dict({})
            for sentence, similarity in results:
                print(f"sentences-------------{sentence}")
                print(f"query-----------------{examples[sentence]}")
                f_ex[sentence] = examples[sentence]


            agent_executor = create_sql_agent(
                llm,
                db=db,
                verbose=True,
                extra_tools=tools
            )
            input_question = f"""YYou are an agent designed to interact with bigquery database.
                Given an input question, create a syntactically correct bigquery query to run.
                
                Return only the final bigquery query and not the results.

                And only use table present in datawhiz dataset and strictly don't use any other tables
                While searching for the tables don't use dataset name before table name example: perfect_order_data
                But while using get_data_dictionary tool and while giving the big query use with dataset name before table name 
                example: datawhiz.perfect_order_data while giving big query only use it when you are mentioning table name
                and not column names

                after selecting the tables use sql_db_schema tool get the schema of the tables and later use get_data_dictionary tool
                You can use get_data_dictionary tool that is provided to you to know about the column meanings in a
                table before starting the solution(it will take table(qualified with a dataset example: dataset.table_name) name as input returns data dictinary of that table)


                Only use the columns present in the tables and don't assume any information.
                Only the SQL query is needed. Dont try to find the result also.
                question : {question}
                Here are some examples of user inputs and their corresponding results required:
                {f_ex}
                Tables must be qualified with a dataset while givig the query(ex: dataset.table_name not only table name) 
                and after getting the query strictly finish the chain don't use sql_db_query_checker multiple times
                """
            st.write("final_examples")
            st.write(f_ex)
            print(f_ex)
            response = agent_executor.invoke({"input": input_question})
            st.write("36")
            return response['output']

        if submit:
            text = generate()
            st.subheader('Query')
            st.write(text)
            test_query = extract_sql_query(text)
            if not test_query:
                test_query = [text]
            if 'limit' not in test_query[0].lower():
                test_query = [test_query[0].replace(";", "") + " LIMIT " + str(no_of_records) + ';']
            QUERY = (test_query[0])
            try:
                query_job = client.query(QUERY)
                rows = query_job.result()
                df = convert_bq_iterator_to_dataframe(rows)
                st.subheader('Result')
                st.write(df)
            except:
                st.write('A valid BigQuery could not be obtained. Please check the question.')

if __name__ == "__main__":
    main()

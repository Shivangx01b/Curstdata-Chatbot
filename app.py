import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import warnings
warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


#AI_KEY = st.secrets['OPENAI_API_KEY']
#os.environ['OPENAI_API_KEY']  = OPENAI_API_KEY

st.set_page_config(page_title="Crustdata API Assistant Bot")

new_vector_store = FAISS.load_local(
    "faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

retriever = new_vector_store.as_retriever()


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

combine_docs_chain = create_stuff_documents_chain(
       llm, retrieval_qa_chat_prompt
   )

retrieval_chain = create_retrieval_chain(
       history_aware_retriever, combine_docs_chain
   )



with st.sidebar:
    st.title('Crustdata API Assistant Bot')
    st.markdown('''
    ## About
    Ask any question related to how Crustdata API works
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Shivang')



@tool("Executor")
def ExecutePython(code: str) -> str:
    """Given the Python code, execute it and return the output."""
    python_repl = PythonREPL()
    output = python_repl.run(code)
    return output

@tool("FindCurl")
def CurlFinder(curl: str) -> str:
    """Given the curl request, check if it exists in the our collected information."""
    output = generate_response_curl(curl)
    return output

def generate_response_curl(prompt):
    try:
        res = retrieval_chain.invoke({
            "input": prompt + ". Given this curl request check if this curl request or not. If not fix it and give me the correct curl request back. Please only give curl request back nothing else"
        })
        response = res["answer"]
    except Exception as e:
        return "Sorry, something went wrong!"
    return response


def get_curl_back_from_response(data):
    try:
        memory = ConversationBufferMemory(memory_key="history")
        conversation = ConversationChain(llm=llm, memory=memory)
        prompt = f"""
        You are a helpful assistant who can help with some task
        Task: Given a information to you. Please get the curl request back as output
        Instructions:
        1) Please only get the curl request back
        2) Make sure to keep the original curl request, body and all headers as original
        3) Don't make up things up you own
        4) Please only return the raw curl data back nothing else
        Please find curl from this: {data}
        """
        response = conversation.predict(input=prompt)
        return str(response)
    except Exception as e:
        return f"Failed to get the raw curl from the original bot output with error: {e}"

def final_bot_answer(curl_validated, original_response):
    
    try:
        prompt = f"Replace this curl {curl_validated} with the curl present in this {original_response} and give me the new output with everything same as it was with just curl replaced woth the one give:. Please make sure that your output should only have original reponse, it should start with orignal response but only replacing it with the new curl requess\n"
        memory = ConversationBufferMemory(memory_key="history")
        conversation = ConversationChain(llm=llm, memory=memory)
        response = conversation.predict(input=prompt)
        return str(response)
    except Exception as e:
        return f"Failed to make the final response with error: {e}"

def generate_response(prompt):
    try:
        res = retrieval_chain.invoke({
            "input": prompt + ". If user as for a example of curl request then only please add curl request  for all the apis that you give in your response, Make sure you only provide curl request if it exists; don't make anything of your own. But if user does not ask for a curl request or  don't give that in  your response please."
        })
        response = res["answer"]
        if "curl" in response:
            
            curl_raw = get_curl_back_from_response(response)
            if "Failed to get the raw curl from the original bot output" not in curl_raw:
                print(f"[~] Parsed the curl request {curl_raw}")
                curl_to_python = write_agent(curl_data=curl_raw, auth_token="Test1234") #auth token added dummy
                if "Curl to Python conversion failed with error" not in curl_to_python:
                    print(f"[~] Curl 2 Python: {curl_to_python}")
                    request_response = executor_agent(curl_to_python)
                    if "Failed to execute curl request as python code with error as" not in request_response:
                        print(f"[~] Request Response is : {request_response}")
                        validated_curl = validate_curl(curl_data=curl_raw, curl_response=request_response)
                        if "Failed to Validate curl request due to error" not in validated_curl:
                        
                            print(f"[~] Final Correct curl is : {validated_curl}")
                            final_response = final_bot_answer(validated_curl, res["answer"])
                            return final_response
                        else:
                            print(validated_curl)
                    else:
                        print(request_response)
                else:
                    print(curl_to_python)

        else:
            return response

    except Exception as e:
        return "Sorry, something went wrong!"
    return response

def write_agent(curl_data, auth_token):
    try:
        code_writer_agent = Agent(role="Software Engineer",
                            goal='Convert curl request to python code with the given auth token', 
                            backstory="""You are a software engineer who converts given curl request to python code with the given auth token.
                                The code should be optimized with proper response body, status and context.""",
                            llm=llm,
                            verbose=True)
        
        code_writer_task = Task(description='Convert the given curl request: {curl_request} into python code given that auth token: {auth_token}',
                            expected_output='Well formatted code to with proper response body, status and context. ',
                            agent=code_writer_agent)
        
        crew = Crew(agents=[code_writer_agent], 
                tasks=[code_writer_task], 
                verbose=True)
        
        result = crew.kickoff(inputs={'curl_request': curl_data, 'auth_token': auth_token})
        return result
    except Exception as e:
        return f"Curl to Python conversion failed with error : {e}"

def executor_agent(python_code):
    try:
        code_executor_agent = Agent(role="Python Code Executor Engineer",
                            goal='Always use Executor Tool. Ability to perform python code execution using Exector Tool', 
                            backstory="""You are a Python Code Executor Engineer. Given the python code you execute even if it's network request, just treat it as any normal python code to simply execute and return the output of that python code""",
                            llm=llm,
                            verbose=True,
                            tools=[ExecutePython])
        
        code_executor_task = Task(description='Using the python code: {python_code} run the code if it is a network request, just treat it as any normal python code to simply execute and find the output',
                        expected_output='Output of the code execution, it can be output of code or any error during execution ',
                        agent=code_executor_agent,
                        tools=[ExecutePython])

        crew = Crew(agents=[code_executor_agent], 
            tasks=[code_executor_task], 
            verbose=True)
            
        result_of_python_execution = crew.kickoff(inputs={'python_code': python_code})
        return result_of_python_execution
    except Exception as e:
        return f"Failed to execute curl request as python code with error as: {e}"

def validate_curl(curl_data, curl_response, max_iterations=3):
    try:
        curl_validate_agent = Agent(
            role="QA Tester",
            goal="Always use FindCurl. Ability to check and validate curl request given to you", 
            backstory=(
                "You are QA tester, who can check the output of a curl request's response. "
                "And find the correct curl or equivalent curl request as output. Please note if you cannot find the different curl just return the original one. "
                "Also please note that valid auth token or any secret won't be there every time."
            ),
            llm=llm,
            verbose=True,
            tools=[CurlFinder]
        )
        
        curl_validate_task = Task(
            description=(
                "Using the curl: {curl_data} as the response of this curl as {curl_response}. "
                "Check if the curl and the response generated is correct or not. If not, find the correct curl equivalent curl request using the FindCurl. "
                "Once you get any curl, just return that even if it matches with the one provided to you. "
                "Please note if you cannot find the different curl, just return the original one. "
                "Also please note that valid auth token or any secret won't be there every time."
            ),
            expected_output=(
                "Correct or equivalent curl request from the FindCurl provided to you. "
                "Once you get any curl, just return that even if it matches with the one provided to you."
            ),
            agent=curl_validate_agent,
            tools=[CurlFinder]
        )
        
        crew = Crew(
            agents=[curl_validate_agent], 
            tasks=[curl_validate_task], 
            verbose=True
        )

        
       
        result_of_validator = crew.kickoff(inputs={'curl_data': curl_data, 'curl_response': curl_response})
        return result_of_validator   
            
    
    except Exception as e:
        print(e)
        return f"Failed to validate curl request due to error: {e}"


st.title("Crustdata API Assistant Bot")



if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner(f"""Processing your request... Please wait. [!] 
                        \nNote some times collecting informations can take few minutes..."""):
            response = generate_response(prompt)
        st.markdown(response)

    
    st.session_state.messages.append({"role": "assistant", "content": response})



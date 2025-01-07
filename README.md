<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/Shivangx01b/Curstdata-Chatbot/blob/main/static/this.PNG" alt="" width="2000px;"></a>
  <br>
  <a href="https://twitter.com/intent/follow?screen_name=shivangx01b"><img src="https://img.shields.io/twitter/follow/shivangx01b?style=flat-square"></a>
</h1>


# Crustdata Build Challenge: Customer Support Agent

# Web App: https://curstdata-chatbot.onrender.com/

## Overview

The task involves building a customer support chatbot that answers questions about Crustdata's APIs. The challenge is divided into three levels, each with progressively complex requirements. Below is a detailed explanation of the task stages and how the provided code achieves these milestones.

---

## Level 0: Basic Static Chat

### Requirements
- **Basic chat interface**: The bot should allow users to ask questions about Crustdata's APIs.
- **Answer technical questions**: Respond with appropriate answers to technical queries.

### Implementation in Code
1. **Chat Interface**:
   - `streamlit` is used to create a simple web app with a sidebar for user interaction.
   - The chat input and messages are managed via `st.session_state` to maintain conversation history.

   ```python
   if "messages" not in st.session_state:
       st.session_state.messages = []

   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])

   if prompt := st.chat_input("What is up?"):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)
   ```

2. **Question Handling**:
   - The `generate_response` function fetches answers to user questions using a retrieval-based QA chain.
   - A FAISS index (`faiss_index`) is used to store and retrieve knowledge about Crustdata APIs.

   ```python
   new_vector_store = FAISS.load_local(
       "faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True
   )
   retriever = new_vector_store.as_retriever()
   ```

3. **Answer Generation**:
   - Questions are processed via the LangChain `retrieval_chain` to provide relevant responses.

   ```python
   retrieval_chain = create_retrieval_chain(
       history_aware_retriever, combine_docs_chain
   )
   response = generate_response(prompt)
   ```

---

## Level 1: Agentic Behavior

### Requirements
- **Validate API requests**: Check the correctness of API examples.
- **Fix API requests**: Use error logs to identify and fix issues.
- **Conversational mode**: Support follow-up questions within a single thread.

### Implementation in Code
1. **API Validation**:
   - The `validate_curl` function ensures that API requests are valid.
   - The validation is performed by comparing the response against expected outputs using tools such as `CurlFinder`.

   ```python
   def validate_curl(curl_data, curl_response, max_iterations=3):
       curl_validate_agent = Agent(
           role="QA Tester",
           goal="Always use FindCurl. Ability to check and validate curl request given to you", 
           tools=[CurlFinder]
       )
       result_of_validator = crew.kickoff(inputs={'curl_data': curl_data, 'curl_response': curl_response})
       return result_of_validator
   ```

2. **Fixing API Requests**:
   - A multi-step pipeline processes raw curl requests, validates them, and fixes errors by generating corrected curl requests using LangChain tools.

   ```python
   def generate_response_curl(prompt):
       res = retrieval_chain.invoke({
           "input": prompt + ". Given this curl request, check if it exists or fix it."
       })
       response = res["answer"]
       return response
   ```

3. **Follow-Up Questions**:
   - The `ConversationBufferMemory` in LangChain allows maintaining context for follow-up questions.

   ```python
   memory = ConversationBufferMemory(memory_key="history")
   conversation = ConversationChain(llm=llm, memory=memory)
   ```

---

## Level 2: Ingestion of Additional Knowledge Base

### Requirements
- **Add a knowledge base**: Incorporate questions and answers from user queries, Slack channels, and updated documentation.
- **Expand API documentation**: Continuously update and improve the dataset.

### Implementation in Code
1. **Knowledge Base Integration**:
   - Additional data sources (e.g., Slack channels and updated documentation) can be indexed and added to the FAISS index.

   ```python
   new_vector_store = FAISS.load_local(
       "faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True
   )
   retriever = new_vector_store.as_retriever()
   ```

2. **Dynamic Updates**:
   - The knowledge base can be re-trained with updated documents or new questions using the same FAISS indexing mechanism.

---

## Features Summary

- **Interactive Chat Interface**: Powered by Streamlit, with real-time user input and bot responses.
- **Retrieval-Based QA**: Uses LangChain and FAISS for accurate responses.
- **API Validation**: Ensures API examples are correct and fixes errors when detected.
- **Conversational Memory**: Supports multi-turn conversations with contextual awareness.
- **Scalable Knowledge Base**: Easy to add new knowledge and update existing data.

---

## Deployment Instructions

1. **Run the App Locally**:
   - Install dependencies: `pip install -r requirements.txt`
   - Start the Streamlit app: `streamlit run app.py`



---

## Ways to improve the code

- Make a better curl validator, cuz current one does not work properly some time , looks like custom tools issue
- Add another agent after Curl to Python to validate the code before run
- Cache the llm response for faster processing
- Move to a better vectordb like Chromadb

  


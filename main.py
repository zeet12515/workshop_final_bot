import streamlit as st
import openai
from langchain.prompts import PromptTemplate
#exercis 12
from langchain.memory import ConversationBufferWindowMemory
#exercise 13
from langchain.document_loaders import TextLoader,PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
import os
import tempfile
#exercise 15
import sqlite3
import pandas as pd
import datetime
#exercise 16
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
#Exercise 17
from langchain.agents import tool
import json
#Exercise 18
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

#Exercise 1: Functions
def ex1():
	st.write("Hello World")

# Exercise 2a : Input , Output and Variables
def ex2():
	name = st.text_input("Enter your name")
	# only prints the Hello {name} if input box is not empty
	if name:
		st.write("Hello " + name)

#Challenge 2 : Input , Output and Variables
def ch2():
	name = st.text_input("Enter your name")
	gender = st.selectbox("State your gender", ["Male", "Female"])
	age = st.text_input("State your age", 18)

	if name and gender and age:
		st.text(f"Hello {name}, you are {gender} and this year you are {age} years old")


#Exercise 3 : Logical Conditioning
def ex3(): 
	age = st.text_input("State your age", 18)
	#if else statement
	age = int(age)
	if age >= 21:
		st.write("You are an adult")
	else:
		st.write("You are not an adult")

#Exercise 4a : Data and Loops 
def ex4a():
	# Data list
	fruits = ["apple", "banana", "orange"]

	# Dictionary
	person = {"name": "John", "age": 30, "city": "New York"}

	# For loop to show list
	st.subheader("Fruits list:")
	for fruit in fruits:
		st.write(fruit)

	#for loop to show dictionary list
	st.subheader("Person dictionary:")
	for key, value in person.items():
		st.write(key + ": " + str(value))

	#Exercise 4b - This part I will ask the participants to put in Exercise 4a or earlier to show the sustained data 
	st.subheader("Session Data:")
	for data in st.session_state.session_data:
		st.write(data)


#Exercise 4a : session_state
def ex4b():

	if "session_data" not in st.session_state:
		st.session_state.session_data = ["alpha", "omega"]

	# For loop to show list
	st.subheader("Session Data:")
	for data in st.session_state.session_data:
		st.write(data)

#Challenge 4 : Data and Loops
def ch4():
	name = st.text_input("Enter your name")
	gender = st.selectbox("State your gender", ["Male", "Female"])
	age = st.text_input("State your age", 18)
	#modify ex2 to set into session_state
	# name = st.session_state.name
	# age = st.session_state.age
	# gender = st.session_state.gender 

	#declare empty dictionary
	mydict = {}
	mydict["name"] = name
	mydict["gender"] = gender
	mydict["age"] = age
	#Print out the items in the dictionary
	st.write("Here is your dictionary: ")
	st.write(mydict)

	#show individual items in dictionary
	st.write("You can also show individual items in the dictionary like this: ")
	for key, value in mydict.items():
		st.write(key + ": " + str(value))

#Exercise 5 : Chatbot UI
def ex5():
	st.title("My first chatbot")

	if "store_msg" not in st.session_state:
		st.session_state.store_msg = []

	prompt = st.chat_input("Say something")
	if prompt:
		st.write(f"User has sent the following prompt: {prompt}")
		st.session_state.store_msg.append(prompt)
		for message in st.session_state.store_msg:
			with st.chat_message("user"):
				st.write(message)
			with st.chat_message("assistant"):
				st.write("Hello human, what can I do for you?")

#Exercise 6 : Rule-based Echo Chatbot 
def ex6():
	st.title("Echo Bot")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What is up?"):
		# Display user message in chat message container
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.messages.append({"role": "user", "content": prompt})

		response = f"Echo: {prompt}"
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(response)
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response})

#Challenge 6 : Rule-based If-Else Chatbot
def ch6():
	st.title("Rule Based Bot")

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("Enter your query"):
		if prompt == "Hello":
			with st.chat_message("user"):
				st.write("Hello")
				st.session_state.messages.append({"role": "user", "content": prompt})
			with st.chat_message("assistant"):
				reply = "Hi there what can I do for you"
				st.write(reply)
				st.session_state.messages.append(
				{"role": "assistant", "content": reply}
				)

		elif prompt == "What is your name?":
			with st.chat_message("user"):
				st.write("What is your name?")
				st.session_state.messages.append({"role": "user", "content": prompt})
			with st.chat_message("assistant"):
				reply = "My name is EAI , an electronic artificial being"
				st.write(reply)
				st.session_state.messages.append(
				{"role": "assistant", "content": reply}
				)

		elif prompt == "How old are you?":
			with st.chat_message("user"):
				st.write("How old are you?")
				st.session_state.messages.append({"role": "user", "content": prompt})
			with st.chat_message("assistant"):
				reply = "Today is my birthday!"
				st.write(reply)
				st.session_state.messages.append(
				{"role": "assistant", "content": reply}
				)

		else:
			with st.chat_message("user"):
				st.write(prompt)
				st.session_state.messages.append({"role": "user", "content": prompt})
			with st.chat_message("assistant"):
				reply = "I am sorry, I am unable to help you with your query"
				st.write(reply)
				st.session_state.messages.append(
				{"role": "assistant", "content": reply}
				)

#Exercise 8 : Using the OpenAI API
def ex8():
	st.title("Api Call")
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo"

	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Tell me about Singapore in the 1970s in 50 words."},
		],
		temperature=0,
	)

	st.markdown("**This is the raw response:**") 
	st.write(response)
	st.markdown("**This is the extracted response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	s = str(response["usage"]["total_tokens"])
	st.markdown("**Total tokens used:**")
	st.write(s)


#Challenge 8: Incorporating the API into your chatbot
def chat_completion(prompt):
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": prompt},
		],
		temperature=0,
	)
	return response["choices"][0]["message"]["content"].strip()

def ch8():
	st.title("My First LLM Chatbot")

	# Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	# Display chat chat_msg from history on app rerun
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	# React to user input
	if prompt := st.chat_input("What's up?"):
		# Display user message in chat message container
		reply = chat_completion(prompt)
		st.chat_message("user").markdown(prompt)
		# Add user message to chat history
		st.session_state.msg.append({"role": "user", "content": prompt})
		# Display assistant response in chat message container
		with st.chat_message("assistant"):
			st.markdown(reply)
		# Add assistant response to chat history
		st.session_state.msg.append({"role": "assistant", "content": reply})


#Exercise 9 : Using the OpenAI API
#Modify exercise 9 - include streaming option
def chat_completion_stream(prompt):
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": prompt},
		],
		temperature= 0, # temperature
		stream=True #stream option
	)
	return response

#integration API call into streamlit chat components
def ex9_basebot():

	#Initialize chat history
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []

	#Showing Chat history
	for message in st.session_state.chat_msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#set user prompt in chat history
			st.session_state.chat_msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.chat_msg.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)

#create a prompt_template session state
#Exercise 10: Basic prompt engineering
def ex10():
	#Step 1 set a new prompt_template session state in def main() 
	#step 2 set it in your message system content
	st.title("Api Call")
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo"
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template },
			{"role": "user", "content": "Tell me about Singapore in the 1970s in 50 words"},
		],
		temperature=0,
	)
	st.markdown("**LLM Response:**")
	st.write(response["choices"][0]["message"]["content"].strip())
	st.markdown("**Total tokens:**")
	st.write(str(response["usage"]["total_tokens"]))

#Challenge 10
#mod chat complete stream function and replace system content to session_state prompt template
def chat_completion_stream_prompt(prompt):
	openai.api_key = st.secrets["openapi_key"]
	MODEL = "gpt-3.5-turbo" #consider changing this to session_state
	response = openai.ChatCompletion.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": st.session_state.prompt_template},
			{"role": "user", "content": prompt},
		],
		temperature= 0, # temperature
		stream=True #stream option
	)
	return response


def ch10_basebot():
  #call the function in your base bot
	#Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	#Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})

	except Exception as e:
		st.error(e)

def ex11():
	#langchain prompt template

	fstring_template = """Tell me a {adjective} story about {content}"""
	prompt = PromptTemplate.from_template(fstring_template)
	final_prompt = prompt.format(adjective="funny", content="chickens")
	st.write(final_prompt)
	pass

def ch11():
	#put in basebot
	fstring_template = """Tell me a {adjective} story about {content}"""
	prompt = PromptTemplate.from_template(fstring_template)
	final_prompt = prompt.format(adjective="funny", content="chickens")
	st.write(final_prompt)
	#set session_state.prompt_template = final_prompt
	st.session_state.prompt_template = prompt.format(adjective="funny", content="chickens")
	#call ch10_basebot
	ch10_basebot()

def ex12():
	memory = ConversationBufferWindowMemory(k=1)
	memory.save_context({"input": "hi"}, {"output": "whats up"})
	memory.save_context({"input": "not much you"}, {"output": "not much"})

	st.write(memory.load_memory_variables({}))
   

	memory = ConversationBufferWindowMemory( k=1, return_messages=True)
	memory.save_context({"input": "hi"}, {"output": "whats up"})
	memory.save_context({"input": "not much you"}, {"output": "not much"})

	st.write(memory.load_memory_variables({}))
	pass

def ch12():
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=5)

	#step 1 save the memory from your chatbot 
	#step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write(memory_data)
	st.session_state.prompt_template = f"""You are a helpful assistant
										This is the last conversation history
										{memory_data}
										"""
	 #call the function in your base bot
	#Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	#Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context({"input": prompt}, {"output": full_response})

	except Exception as e:
		st.error(e)
	pass

#exercise 13 - loading
#pip install pypdf
#pip install lancedb
def upload_file_streamlit():

	def get_file_extension(file_name):
		return os.path.splitext(file_name)[1]

	st.subheader("Upload your docs")

	# Streamlit file uploader to accept file input
	uploaded_file = st.file_uploader("Choose a file", type=['docx', 'txt', 'pdf'])

	if uploaded_file:

		# Reading file content
		file_content = uploaded_file.read()

		# Determine the suffix based on uploaded file's name
		file_suffix = get_file_extension(uploaded_file.name)

		# Saving the uploaded file temporarily to process it
		with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
			temp_file.write(file_content)
			temp_file.flush()  # Ensure the data is written to the file
			temp_file_path = temp_file.name
		return temp_file_path
	
#exercise 13 - split and chunk, embeddings and storing in vectorstores for reference
def vecstore_creator(query):
	if "vectorstore" not in st.session_state:
		st.session_state.vectorstore = False
	
	os.environ['OPENAI_API_KEY'] = st.secrets["openapi_key"]
	# Process the temporary file using UnstructuredFileLoader (or any other method you need)
	embeddings = OpenAIEmbeddings()
	db = lancedb.connect("/tmp/lancedb")
	table = db.create_table(
		"my_table",
		data=[
			{
				"vector": embeddings.embed_query("Hello World"),
				"text": "Hello World",
				"id": "1",
			}
		],
		mode="overwrite",
	)
	#st.write(temp_file_path)
	temp_file_path = upload_file_streamlit()
	if temp_file_path:
		loader = PyPDFLoader(temp_file_path )
		documents = loader.load_and_split()
		vectorestore = LanceDB.from_documents(documents, OpenAIEmbeddings(), connection=table)
		st.session_state.vectorstore = vectorestore
		if query:
			docs = vectorestore.similarity_search(query)
			st.write(docs[0].page_content)

def ex13_vectorstore_creator():
	query = st.text_input("Enter a query")
	vecstore_creator(query)

#save the vectorstore in st.session_state
#add semantic search prompt into memory prompt
#integrate back into your chatbot
def ex14():

	vecstore_creator(False)
	
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=5)

	#step 1 save the memory from your chatbot 
	#step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write(memory_data)
	st.session_state.prompt_template = f"""You are a helpful assistant
										This is the last conversation history
										{memory_data}
										"""
	 #call the function in your base bot
	#Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	#Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#query information
			if st.session_state.vectorstore:
				docs = st.session_state.vectorstore.similarity_search(prompt)
				docs = docs[0].page_content 
				#add your query prompt
				vs_prompt = f"""You should reference this search result to help your answer,
								{docs}
								if the search result does not anwer the query, please say you are unable to answer, do not make up an answer"""
			else:
				vs_prompt = ""
			#add query prompt to your memory prompt and send it to LLM
			st.session_state.prompt_template = st.session_state.prompt_template + vs_prompt
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context({"input": prompt}, {"output": full_response})

	except Exception as e:
		st.error(e)

#collecting data using sql server
def ex15():
	# Create or check for the 'database' directory in the current working directory
	cwd = os.getcwd()
	database_path = os.path.join(cwd, "database")

	if not os.path.exists(database_path):
		os.makedirs(database_path)

	# Set DB_NAME to be within the 'database' directory
	DB_NAME = os.path.join(database_path, "default_db")

	# Connect to the SQLite database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Conversation data table
	cursor.execute('''
		CREATE TABLE IF NOT EXISTS data_table (
			id INTEGER PRIMARY KEY,
			date TEXT NOT NULL UNIQUE,
			username TEXT NOT NULL,
			chatbot_ans TEXT NOT NULL,
			user_prompt TEXT NOT NULL,
			tokens TEXT
		)
	''')
	conn.commit()
	conn.close()

#implementing data collection and displaying 
def ex15_display():
#display data
	cwd = os.getcwd()
	database_path = os.path.join(cwd, "database")

	if not os.path.exists(database_path):
		os.makedirs(database_path)

	# Set DB_NAME to be within the 'database' directory
	DB_NAME = os.path.join(database_path, "default_db")
	# Connect to the specified database
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	# Fetch all data from data_table
	cursor.execute("SELECT * FROM data_table")
	rows = cursor.fetchall()
	column_names = [description[0] for description in cursor.description]
	df = pd.DataFrame(rows, columns=column_names)
	st.dataframe(df)
	conn.close()

def ex15_collect(username, chatbot, prompt):
#collect data from bot
	cwd = os.getcwd()
	database_path = os.path.join(cwd, "database")

	if not os.path.exists(database_path):
		os.makedirs(database_path)

	# Set DB_NAME to be within the 'database' directory
	DB_NAME = os.path.join(database_path, "default_db")
	conn = sqlite3.connect("database")
	cursor = conn.cursor()
	now = datetime.now() # Using ISO format for date
	tokens = len(chatbot)*1.3
	cursor.execute('''
		INSERT INTO data_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?, ?)
	''', (now, username, chatbot, prompt, tokens))
	conn.commit()
	conn.close()

def ch15_chatbot():
	vecstore_creator(False)
	ex15_display()
	if "memory" not in st.session_state:
		st.session_state.memory = ConversationBufferWindowMemory(k=5)

	#step 1 save the memory from your chatbot 
	#step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
	memory_data = st.session_state.memory.load_memory_variables({})
	st.write(memory_data)
	st.session_state.prompt_template = f"""You are a helpful assistant
										This is the last conversation history
										{memory_data}
										"""
	 #call the function in your base bot
	#Initialize chat history
	if "msg" not in st.session_state:
		st.session_state.msg = []

	#Showing Chat history
	for message in st.session_state.msg:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
	try:
		#
		if prompt := st.chat_input("What is up?"):
			#query information
			if st.session_state.vectorstore:
				docs = st.session_state.vectorstore.similarity_search(prompt)
				docs = docs[0].page_content
				#add your query prompt
				vs_prompt = f"""You should reference this search result to help your answer,
								{docs}
								if the search result does not anwer the query, please say you are unable to answer, do not make up an answer"""
			else:
				vs_prompt = ""
			#add query prompt to your memory prompt and send it to LLM
			st.session_state.prompt_template = st.session_state.prompt_template + vs_prompt
			#set user prompt in chat history
			st.session_state.msg.append({"role": "user", "content": prompt})
			with st.chat_message("user"):
				st.markdown(prompt)

			with st.chat_message("assistant"):
				message_placeholder = st.empty()
				full_response = ""
				#streaming function
				for response in chat_completion_stream_prompt(prompt):
					full_response += response.choices[0].delta.get("content", "")
					message_placeholder.markdown(full_response + "▌")
				message_placeholder.markdown(full_response)
			st.session_state.msg.append({"role": "assistant", "content": full_response})
			st.session_state.memory.save_context({"input": prompt}, {"output": full_response})
			#collect data
			ex15_collect("username", full_response, prompt)

	except Exception as e:
		st.error(e)

#smart agents accessing the internet for free
#https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/search_and_chat.py
def ex16():
	st.title("🦜 LangChain: Chat with search")

	openai_api_key = st.secrets["openapi_key"]

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
		tools = [DuckDuckGoSearchRun(name="Search")]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

#agents ,vectorstores, wiki 
#https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
#note tool
@tool("Document search")
def document_search(query: str) ->str:
	"Use this function first to search for documents pertaining to the query before going into the internet"
	docs = st.session_state.vectorstore.similarity_search(query)
	docs = docs[0].page_content
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

def ex17():
	vecstore_creator(False)
	#st.session_state.vectorstore

	st.title("🦜 LangChain: Chat with search")

	openai_api_key = st.secrets["openapi_key"]

	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
		tools = [DuckDuckGoSearchRun(name="Internet Search"), document_search]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

#Pandai - A smart agent that can do visual analytics
def ex18():
	st.title("pandas-ai streamlit interface")

	# Upload CSV file using st.file_uploader
	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if "openai_key" not in st.session_state:
		st.session_state.openai_key = st.secrets["openapi_key"]
		st.session_state.prompt_history = []
		st.session_state.df = None
	
	if st.session_state.df is None:
		# If a file is uploaded, read it with pandas and display the DataFrame
		if uploaded_file is not None:
			try:
				df = pd.read_csv(uploaded_file)
				st.session_state.df = df
			except Exception as e:
				st.write("There was an error processing the CSV file.")
				st.write(e)

	# Check if df is a DataFrame instance
	if  st.session_state.df is None:
		st.session_state.df = pd.DataFrame({
			"country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
			"gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
			"happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
		})
	
	with st.form("Question"):
		question = st.text_input("Question", value="", type="default")
		submitted = st.form_submit_button("Submit")
		if submitted:
			with st.spinner():
				llm = OpenAI(api_token=st.session_state.openai_key)
				df = SmartDataframe(st.session_state.df, config={"llm": llm})
				response = df.chat(question)  # Using 'chat' method based on your context.
            
				# After generating the chart (if applicable), display it:
				chart_path = os.path.join("exports/charts", "temp_chart.png")
				if os.path.exists(chart_path):
					plt.savefig(chart_path)
					st.image(chart_path, caption="Generated Chart", use_column_width=True)
				
				# Display the textual response (if any):
				if response:
					st.write(response)
				
				# Append the question to the history:
				st.session_state.prompt_history.append(question)

	if st.session_state.df is not None:
		st.subheader("Current dataframe:")
		st.write(st.session_state.df)

	st.subheader("Prompt history:")
	st.write(st.session_state.prompt_history)

	if st.button("Clear"):
		st.session_state.prompt_history = []
		st.session_state.df = None


	


#Exercise 2b : creating st.sidebar
def main():

	#ex4b
	if "session_data" not in st.session_state:
		st.session_state.session_data = ["alpha", "omega"]
	#challenge 4b
	if "age" not in st.session_state:
		st.session_state.age = 0
	
	if "name" not in st.session_state:
		st.session_state.name = ""
	
	if "gender" not in st.session_state:
		st.session_state.gender = ""
	#ex10
	if "prompt_template" not in st.session_state:
		st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"
#Exercise 2b:
	# with st.sidebar:
	# 	st.write("Side bar test")
	
	# ex1()
	# ex2()
	# ch2()
	# ex3()

#Challenge 3
	with st.sidebar:
		option = st.selectbox("Function exercises", ["ex1", "ex2", "ch2", "ex3", "ch3", "ex4a", "ex4b", "ch4", "ex5", "ex6", "ch6", "ex8", "ch8", "ex9", "ex10", "ch10", "ex11", "ch11", "ex12", "ch12", "ex13", "ex14", "ex15", "ch15", "ex16", "ex17", "ex18"])
	if option == "ex1":
		ex1()
	elif option == "ex2":
		ex2()
	elif option == "ch2":
		ch2()
	elif option == "ex3":
		ex3()
#end of challenge 3
	elif option == "ex4a":
		ex4a()
	elif option == "ex4b":
		ex4b()
	elif option == "ch4":
		ch4()
	elif option == "ex5":
		ex5()
	elif option == "ex6":
		ex6()
	elif option == "ch6":
		ch6()
	elif option == "ex8":
		ex8()
	elif option == "ch8":
		ch8()
	elif option == "ex9":
		ex9_basebot()
	elif option == "ex10":
		ex10()
	elif option == "ch10":
		ch10_basebot()
	elif option == "ex11":
		ex11()
	elif option == "ch11":
		ch11()
	elif option == "ex12":
		ex12()
	elif option == "ch12":
		ch12()
	elif option == "ex13":
		ex13_vectorstore_creator()
	elif option == "ex14":
		ex14()
	elif option == "ex15":
		ex15()
	elif option == "ch15":
		ch15_chatbot()
	elif option == "ex16":
		ex16()
	elif option == "ex17":
		ex17()
	elif option == "ex18":
		ex18()
	
	
	
if __name__ == "__main__":
	main()


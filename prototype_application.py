import streamlit as st
from main_bot import basebot
from kb_module import display_vectorstores
from users_module import vectorstore_selection_interface

if "form_title" not in st.session_state:
	st.session_state.form_title = "Bot Persona"
if "question_1" not in st.session_state:
	st.session_state.question_1 = "Name"
if "question_2" not in st.session_state:
	st.session_state.question_2 = "Occupation"
if "question_3" not in st.session_state:
	st.session_state.question_3 = "Subject"
if "question_4" not in st.session_state:
	st.session_state.question_4 = "Required Materials"
if "question_5" not in st.session_state:
	st.session_state.question_5 = "Others"
if "my_app_template" not in st.session_state:
	st.session_state.my_app_template = "Pretend you are a {q2}, I want you to be an expert on the subject {q3} , the student is {q1}"

def form_input():
	with st.form("my_form"):
		st.subheader(st.session_state.form_title)
		q1 = st.text_input(st.session_state.question_1)
		q2 = st.text_input(st.session_state.question_2)
		q3 = st.text_input(st.session_state.question_3)
		q4 = st.text_input(st.session_state.question_4)
		q5 = st.text_input(st.session_state.question_5)

		# Every form must have a submit button.
		submitted = st.form_submit_button("Submit")
		if submitted:
			return q1, q2, q3, q4, q5
		
	return False

def form_settings():
	title = st.text_input("Form Title", value=st.session_state.form_title)
	question_1 = st.text_input("Question 1:", value=st.session_state.question_1)
	question_2 = st.text_input("Question 2:", value=st.session_state.question_2)
	question_3 = st.text_input("Question 3:", value=st.session_state.question_3)
	question_4 = st.text_input("Question 4:", value=st.session_state.question_4)
	question_5 = st.text_input("Question 5:", value=st.session_state.question_5)
	if st.button("Update Questions"):
		st.session_state.form_title = title
		st.session_state.question_1 = question_1
		st.session_state.question_2 = question_2
		st.session_state.question_3 = question_3
		st.session_state.question_4 = question_4
		st.session_state.question_5 = question_5

def chatbot_settings():
	temp = st.number_input("Temperature", value=st.session_state.temp, min_value=0.0, max_value=1.0, step=0.1)
	k_memory = st.number_input("K Memory", value=st.session_state.k_memory, min_value=0, max_value=4, step=1)
	if st.button("Update Chatbot Settings"):
		st.session_state.temp = temp
		st.session_state.k_memory = k_memory


def prompt_template_settings():
	st.info("You can use the following variables which is link to your first 5 questions in your prompt template: {q1}, {q2}, {q3}, {q4}, {q5}")
	prompt_template = st.text_area("Enter your prompt template here:", value = st.session_state.my_app_template )
	if st.button("Update Prompt Template"):
		st.session_state.my_app_template = prompt_template


def prompt_template(results):
	return st.session_state.my_app_template.format(q1=results[0], q2=results[1], q3=results[2], q4=results[3], q5=results[4])

def my_first_app(bot_name):
	st.subheader("Protyping a chatbot")
	results = form_input()
	if results != False:
		st.session_state.chatbot = prompt_template(results)
		st.write(st.session_state.chatbot)
	basebot(bot_name)

def prototype_settings():
	tab1, tab2, tab3, tab4 = st.tabs(["Prototype Input Settings", "Template settings", "Prototype Chatbot Settings", "KB settings"])

	with tab1:
		st.subheader("Prototype Input and template Settings")
		form_settings()
		prompt_template_settings()

	with tab2:
		st.subheader("Template settings")
		
		
	with tab3:
		st.subheader("Prototype Chatbot Settings")
		chatbot_settings()

	with tab4:
		st.subheader("KB settings")
		st.write("KB settings")
		display_vectorstores()
		vectorstore_selection_interface(st.session_state.user['id'])

	
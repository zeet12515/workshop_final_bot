import openai
import streamlit as st
from authenticate import return_api_key
import os

def call_api():
	prompt_design = st.text_input("Enter your the prompt design for the API call:", value="You are a helpful assistant.")
	prompt_query = st.text_input("Enter your prompt query:", value="Tell me about Singapore in the 1970s in 50 words.")
	if st.button("Call the API"):
		if prompt_design and prompt_query:
			api_call(prompt_design, prompt_query)
		else:
			st.warning("Please enter a prompt design and prompt query.")
	

def api_call(p_design, p_query):
	openai.api_key = return_api_key()
	os.environ["OPENAI_API_KEY"] = return_api_key()
	st.title("Api Call")
	MODEL = "gpt-3.5-turbo"
	with st.status("Calling the OpenAI API..."):
		response = openai.ChatCompletion.create(
			model=MODEL,
			messages=[
				{"role": "system", "content": p_design},
				{"role": "user", "content": p_query},
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
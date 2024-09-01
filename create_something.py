import streamlit as st
from streamlit_chat import message
import dashscope
from dashscope import Application
import uuid

import streamlit as st
from streamlit_chat import message
import dashscope
from dashscope import Application
import uuid

def call_agent_app(prompt):
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    response = Application.call(
        app_id=st.secrets['QWEN_API']['APP_ID'],
        prompt=prompt,
        api_key=st.secrets['QWEN_API']['API_KEY'],
    )
    return response.output.text

def main():
    # Initialize an empty message state if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Clear previous messages to display only the latest ones
        st.session_state['messages'] = []

        # Append the user message to the session state
        st.session_state['messages'].append({"text": user_input, "is_user": True})

        # Placeholder for loading gif while fetching the response
        with st.spinner("Waiting for response..."):
            response_text = call_agent_app(user_input)
        
        # Append the AI response to the session state
        st.session_state['messages'].append({"text": response_text, "is_user": False})

    # Display only the latest user input and AI response
    if st.session_state['messages']:
        user_message = st.session_state['messages'][0]
        ai_message = st.session_state['messages'][1]

        with st.container():
            message(
                user_message['text'], 
                is_user=True, 
                key=f"{uuid.uuid1()}_user"
            )
            message(
                ai_message['text'], 
                is_user=False, 
                key=f"{uuid.uuid1()}_generated"
            )


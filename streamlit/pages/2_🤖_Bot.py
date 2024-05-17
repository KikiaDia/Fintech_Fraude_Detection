import streamlit as st

from langserve import RemoteRunnable

llm = RemoteRunnable("http://localhost:8000/bot/")

st.set_page_config(
  page_title='Bot',
  layout='wide',
  page_icon='📜'
)

# # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
resp = llm.invoke({"question": prompt},
                        config={"configurable": {"session_id": "123"}},)
    
response = resp.content
# print(prompt)
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
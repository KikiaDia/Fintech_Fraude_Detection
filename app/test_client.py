from langserve import RemoteRunnable

ollama_llm = RemoteRunnable("http://localhost:8000/bot/")


# resp = ollama_llm.invoke({"question": "quel est le nombre de transactions de tyoe P2P?"},
#                         config={"configurable": {"session_id": ""}},
# ) 
resp = ollama_llm.invoke({"question": "quel est le nombre de transactions a l'utilisateur avec l'id 18222 etait le destinataire?"},
                        config={"configurable": {"session_id": ""}},
) 
print(resp.content)
from langserve import RemoteRunnable

llm = RemoteRunnable("http://localhost:8000/bot/")


# resp = llm.invoke({"question": "quel est le nombre de transactions de type P2P?"},
#                         config={"configurable": {"session_id": ""}},
# ) 
# resp = llm.invoke({"question": "quel est le nombre de transactions a l'utilisateur avec l'id 18222 etait le destinataire?"},
#                         config={"configurable": {"session_id": ""}},
# ) 
# resp = llm.invoke({"question": "le montant total de transaction pour le type CASHIN  et CASHOUT?"},
#                         config={"configurable": {"session_id": ""}},
# ) 

resp = llm.invoke({"question": "quel sont les types de fraude en fintech?"},
                        config={"configurable": {"session_id": ""}},)
print(resp)
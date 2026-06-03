from langchain_core.prompts import ChatPromptTemplate

# ✅ Prompt court = moins de tokens d'entrée = moins de latence
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Réponds à la question avec le contexte. Sois concis et précis."),
    ("human", "{question}\n\nContexte:\n{context}")
])
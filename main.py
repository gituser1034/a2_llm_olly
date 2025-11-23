# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

system_prompt_path = 'prompts/prompt.txt'

# Getting system prompt
with open(system_prompt_path, 'r', encoding='utf-8') as file:
  system_prompt = file.read().strip()

# Potential issue with seed generating same question each time, may need to 
# have seed as a variable and increment each time in loop??
model = OllamaLLM(
    model="qwen3:4b",
    seed=1
)

# Placeholder for read in file later
# template = """
# You are an expert in answering questions about a pizza restaurant.

# Here are some relevant reviews: {reviews}

# Here is the question to answer: {question}
# """

prompt = ChatPromptTemplate.from_template(system_prompt)
chain = prompt | model

# Should there be a model introduction first?
print("Welcome to the Ollama LLM Astronomy Quiz.")

# This is user asks question, then model grabs info relating to question
# We want model to go into vector store based on prompt, and model finds a question and answer
# Model asks question, user answers, model checks based on answer and repeats
while True:

    q = retriever.invoke("Generate Astronomy questions based on the given notes")

    user_input = input("Answer the question (q to quit): ")
    if user_input == "q":
        break
    
    # Getting data from document store
    reviews = retriever.invoke(user_input)
    result = chain.invoke({"reviews": reviews, "answer": user_input})
    print(result)
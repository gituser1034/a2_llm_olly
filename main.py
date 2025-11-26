# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever

system_prompt_path = 'prompts/prompt.txt'

# Big issue, w 2 chains the llm generating 15 quiz questions, and answering them before user can answer anything
# chains working wrong? Incorrect order? Should find resource for langchain ollama q+a
# Did I just word the prompt badly so it thinks it needs to generate 15 questions every time?

# First just get RAG with question generation working, then add user response
# and figure out multiple chains

# Getting system prompt
with open(system_prompt_path, 'r', encoding='utf-8') as file:
  system_prompt = file.read().strip()

# Potential issue with seed generating same question each time, may need to 
# have seed as a variable and increment each time in loop??
model = OllamaLLM(
    model="qwen3:4b",
    #seed=1
)

question_prompt_path = 'prompts/q_prompt.txt'
answer_prompt_path = 'prompts/a_prompt.txt'
# Getting question prompt
with open(question_prompt_path, 'r', encoding='utf-8') as file:
  question_prompt = file.read().strip()

with open(answer_prompt_path, 'r', encoding='utf-8') as file:
  answer_prompt = file.read().strip()

# Placeholder for read in file later
# template = """
# You are an expert in answering questions about a pizza restaurant.

# Here are some relevant reviews: {reviews}

# Here is the question to answer: {question}
# """

# Need 2 prompts and 2 chains, one for generating questions, one for answering
# later need to figure out how these tie into one main rules/system prompt
# prompt = ChatPromptTemplate.from_template(system_prompt)
# chain = prompt | model

prompt = ChatPromptTemplate.from_template(question_prompt)
chain = prompt | model
prompt2 = ChatPromptTemplate.from_template(answer_prompt)
chain2 = prompt2 | model

# Should there be a model introduction first?
print("Welcome to the Ollama LLM Astronomy Quiz.")

while True:
    # Searches through vector store to generate a question
    q_data = retriever.invoke("Generate an astronomy quiz question")

    # Generating question based on retrieved data
    q = chain.invoke({"q_data": q_data})
    print("Question: ", q)

    # Need question generated first
    user_input = input("Respond (q to quit): ")
    if user_input == "q":
        break

    answer = retriever.invoke(user_input)
    result = chain2.invoke({"answer": answer, "user_input": user_input})
    print(result)
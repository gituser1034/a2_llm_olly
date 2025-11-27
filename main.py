# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever

system_prompt_path = 'prompts/prompt.txt'

# Retrieval very slow
# Change pipeline so that llm goes into document, generates 15 questions
# and answers - this will be seeded so its the 
# same set of 15 each time for later evaluation
# then user will respond and llm will evaluate that response based on 
# the already known answers - it refers to its answer list for the certain index and re-explains

# Getting system prompt
# with open(system_prompt_path, 'r', encoding='utf-8') as file:
#   system_prompt = file.read().strip()

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

# Getting prompt for students answer
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

# 1 chain using RAG for the model to generate a question
# another chain for the model to respond to the user
prompt = ChatPromptTemplate.from_template(question_prompt)
chain = prompt | model
prompt2 = ChatPromptTemplate.from_template(answer_prompt)
chain2 = prompt2 | model

# Should there be a model introduction first?
print("Welcome to the Ollama LLM Astronomy Quiz.")

# Tracking quiz rounds - will I use this?
i = 0

while True:
    # Retrieves notes to use in generation
    astro_notes = retriever.invoke("Ask an astronomy question")
    output = chain.invoke({"astro_notes": astro_notes})
    print(output)

    print("---------------------------")

    student_answer = input("Answer the question (q to quit): ")

    if student_answer == "q":
      break

    astro_notes = retriever.invoke(student_answer)
    output = chain2.invoke({"astro_notes": astro_notes, "student_answer": student_answer})
    print(output)

    print("---------------------------")

    #i+=1
    
    # Searches through vector store to generate a question
    # q_data = retriever.invoke("Generate an astronomy quiz question")

    # # Generating question based on retrieved data
    # q = chain.invoke({"q_data": q_data})
    # print("Question: ", q)

    # # Need question generated first
    # user_input = input("Respond (q to quit): ")
    # if user_input == "q":
    #     break

    # answer = retriever.invoke(user_input)
    # result = chain2.invoke({"answer": answer, "user_input": user_input})
    # print(result)
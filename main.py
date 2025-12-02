# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever

system_prompt_path = 'prompts/prompt.txt'

# Retrieval very slow - new pipeline
# LLM goes into document, generates 15 questions
# These questions get stored in a list and each iteration will be read to the user
# user will type in an answer, and model will use RAG again to retrieve answer from the document and 
# give a 1 or 0 grade, this will be grabbed from the output to give the user a final grade

# Seeding not working, will need to change how this works
# Have document of questions that model grabs from, reads in, and asks?
# For reproducibility and output checking
# or change from study budy to me quizzing the model


# Getting system prompt
# with open(system_prompt_path, 'r', encoding='utf-8') as file:
#   system_prompt = file.read().strip()

# Potential issue with seed generating same question each time, may need to 
# have seed as a variable and increment each time in loop??
model = OllamaLLM(
  model="qwen3:4b",
  options={"seed": 42}
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

questions = []

# Retrieves notes to use in generation
astro_notes = retriever.invoke("Ask astronomy questions")
output = chain.invoke({"astro_notes": astro_notes})
print(output)
print("---------------------------")

# Want to split output on newline and ignore index 0
questions = output.split("\n")

for i in range(len(questions)):
  print(questions[i])
  print("---------------------------")

  student_answer = input("Answer the question (q to quit): ")

  if student_answer == "q":
    break

  astro_notes = retriever.invoke(student_answer)
  output = chain2.invoke({"astro_notes": astro_notes, "student_answer": student_answer})
  print(output)

  print("---------------------------")

# while True:

#     # Outputs like:
#     # Welcome to the Ollama LLM Astronomy Quiz.
#     # 1. What is the key difference between a sidereal day and a solar day?  
#     # 2. Why do astronomers prefer using sidereal days for calculations?  
#     # 3. What does apparent solar time depend on for its measurement?  

#     print("---------------------------")

#     student_answer = input("Answer the question (q to quit): ")

#     if student_answer == "q":
#       break

#     # astro_notes = retriever.invoke(student_answer)
#     # output = chain2.invoke({"astro_notes": astro_notes, "student_answer": student_answer})
#     # print(output)

#     print("---------------------------")

#     #i+=1
    
#     # Searches through vector store to generate a question
#     # q_data = retriever.invoke("Generate an astronomy quiz question")

#     # # Generating question based on retrieved data
#     # q = chain.invoke({"q_data": q_data})
#     # print("Question: ", q)

#     # # Need question generated first
#     # user_input = input("Respond (q to quit): ")
#     # if user_input == "q":
#     #     break

#     # answer = retriever.invoke(user_input)
#     # result = chain2.invoke({"answer": answer, "user_input": user_input})
#     # print(result)
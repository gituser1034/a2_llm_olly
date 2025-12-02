# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever

system_prompt_path = 'prompts/prompt.txt'

# FINISHED THIS PART - Test
# Better idea for reducing speed of RAG - NEED TO IMPLEMENT
# model uses RAG to generate 15 questions and answers, questions stored in list like
# current and output 1 by 1 from the list, answers are stored in some text string
# like answer then newline the model has access to, a "local cache", 
# So when user asks model question, it responds with context from local cache
# May do a true or false test for easy answer equating in test

# How to test
# Offline testing - Im not testing the whole flow, just the grading part
# I have a test file thats like this but slightly modified
# I provide the model the questions all at once, 15 inputs with answers
# It uses RAG going through document, it generates all answers at once
# i split them and store in a list, 1 answer per line
# I equate these to the expected answers 
# JUST JUDGING HOW IT USES RAG TO GRADE! NOT WHOLE FLOW! IN README EXPLAIN WHY THAT WOULD 
# BE IMPOSSIBLE!


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
results = []
models_answers = ""

# Retrieves notes to use in generation
astro_notes = retriever.invoke("Ask astronomy questions and provide answers")
output = chain.invoke({"astro_notes": astro_notes})
print(output)
print("---------------------------")

results = output.split("\n")

# accessing first index and checking whether its Q or A
# to seperately store questions and answers
for i in range(len(results)):
  # Checking first char in the line
  if results[i][0] == "Q":
    questions.append(results[i])
  elif results[i][0] == "A":
    models_answers += (results[i] + "\n")

for i in range(len(questions)):
  print(questions[i])
  print("---------------------------")

  student_answer = input("Answer the question (q to quit): ")

  if student_answer == "q":
    break

  # What do i pass here
  output = chain2.invoke("Evaluate the students answer")
  print(output)

  # Want to replace this rag with regular chatbot loop
  # astro_notes = retriever.invoke(student_answer)
  # output = chain2.invoke({"astro_notes": astro_notes, "student_answer": student_answer})
  # print(output)

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
# Below in venv
# Im using local llm so may have to modify slightly
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever
from pydantic import BaseModel
from typing import List

system_prompt_path = 'prompts/prompt.txt'

# How to test
# Offline testing - Im not testing the whole flow, just the grading part
# I have a test file thats like this but slightly modified
# I provide the model the questions all at once, 15 inputs with answers
# It uses RAG going through document, it generates all answers at once
# i split them and store in a list, 1 answer per line
# I equate these to the expected answers 
# JUST JUDGING HOW IT USES RAG TO GRADE! NOT WHOLE FLOW! IN README EXPLAIN WHY THAT WOULD 
# BE IMPOSSIBLE!

model = OllamaLLM(
  model="qwen3:4b",
)

question_prompt_path = 'prompts/q_prompt.txt'
answer_prompt_path = 'prompts/a_prompt.txt'

# Getting question prompt
with open(question_prompt_path, 'r', encoding='utf-8') as file:
  question_prompt = file.read().strip()

# Getting prompt for students answer
with open(answer_prompt_path, 'r', encoding='utf-8') as file:
  answer_prompt = file.read().strip()

# 1 chain using RAG for the model to generate a question
# another chain for the model to respond to the user
prompt = ChatPromptTemplate.from_template(question_prompt)
chain = prompt | model
prompt2 = ChatPromptTemplate.from_template(answer_prompt)
chain2 = prompt2 | model

print("Welcome to the Ollama LLM Astronomy True or False Quiz.")

# Local caches - Initial RAG generation data stored here for
# ideally faster chat 
results = []
questions = []
models_answers = []

# Retrieves notes to use in generation
astro_notes = retriever.invoke("Ask astronomy questions and provide answers")
output = chain.invoke({"astro_notes": astro_notes})
results = output.split("\n")

# accessing first index and checking whether its Q or A
# to seperately store questions and answers
for i in range(len(results)):
  line = results[i].strip()
  if not line:
    continue
  # Checking first char in the line
  if line[0] == "Q":
    questions.append(line)
  elif line[0] == "A":
    models_answers.append(line + "\n")
  else:
    continue

for i in range(len(questions)):
  print(questions[i])
  print("---------------------------")

  student_answer = input("Answer the question (q to quit): ")

  if student_answer == "q":
    break

  # Passing context the answer prompt expects
  output = chain2.invoke({"models_answer": models_answers[i], "question": questions[i], "student_answer":student_answer})
  print(output)

  print("---------------------------")
# Testing LLM Grading ability
# Using questions generated in a previous round to test models grading ability

# Below in venv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorpdf import retriever
from pydantic import BaseModel
from typing import List
import datetime
import time
import json

telemetry = 'samples/telemetry_test.txt'

model = OllamaLLM(
  model="qwen3:4b",
)

test_prompt_path = 'prompts/test_prompt.txt'
answer_prompt_path = 'prompts/a_prompt.txt'
inputs_path = 'prompts/inputs.json'
inputs = []
questions = []
expected_answers = []
# Tracks how many questions correct
acc_count = 0
model_acc_count = 0

# Reading in inputs from json file
with open(inputs_path, 'r', encoding='utf-8') as file:
  data = json.load(file)

for input_key in data:
  inputs.append(data[input_key]["input"])
  questions.append(data[input_key]["question"])
  expected_answers.append(data[input_key]["expected"])

# Getting question prompt
with open(test_prompt_path, 'r', encoding='utf-8') as file:
  test_prompt = file.read().strip()

# Getting prompt for students answer
with open(answer_prompt_path, 'r', encoding='utf-8') as file:
  answer_prompt = file.read().strip()

# 1 chain using RAG for the model to generate a question
# another chain for the model to respond to the user
prompt = ChatPromptTemplate.from_template(test_prompt)
chain = prompt | model
prompt2 = ChatPromptTemplate.from_template(answer_prompt)
chain2 = prompt2 | model

print("Welcome to the Ollama LLM Astronomy True or False Quiz.")

# Local caches - Initial RAG generation data stored here for
# ideally faster chat 
results = []
models_answers = []

with open(telemetry, 'a', encoding='utf-8') as file:
  # From the models previously generated set of questions I have stored with my test cases
  # Generate answers stored in a cache for later grading
  file.write(str(datetime.datetime.now()) + ": RAG retrieval underway - accessing AstroNotes.pdf chroma_langchain_db vector store\n")
  start_time = time.time()
  output = chain.invoke({"test_questions": questions})
  end_time = time.time()
  file.write(str(datetime.datetime.now()) + ": RAG data retrieval successful\n")
  file.write("RAG Latency: " + str(end_time - start_time)  + "\n")

  file.write(str(datetime.datetime.now()) + ": Caching data.\n")
  start_time2 = time.time()
  results = output.split("\n")

  # Storing model answers
  for i in range(len(results)):
    line = results[i].strip()
    if not line:
      continue
    if line[0] == "A":
      models_answers.append(line + "\n")

  file.write(str(datetime.datetime.now()) + ": Data cached.\n")
  end_time2 = time.time()
  file.write("Caching Latency: " + str(end_time2 - start_time2) + "\n")

  for i in range(len(inputs)):
    print(questions[i])
    print("---------------------------")
    file.write(str(datetime.datetime.now()) + ": Cached question output: "  + questions[i] + "\n")

    print(f"Student: {inputs[i]}\n")
    file.write(str(datetime.datetime.now()) + ": Student: " + inputs[i] + "\n")

    if inputs[i] == "q":
      break

    file.write(str(datetime.datetime.now()) + ": Local cache retrieval underway.\n")
    start_time3 = time.time()
    # Passing context the answer prompt expects
    # Error - 15 model answers, 2 extra test cases - just delete these and were good?
    output = chain2.invoke({"models_answer": models_answers[i], "question": questions[i], "student_answer":inputs[i]})
    end_time3 = time.time()
    file.write(str(datetime.datetime.now()) + ": Cached data retrieval successful\n")
    file.write(str(datetime.datetime.now()) + ": Model: " + output  + "\n")
    file.write("Cache retrieval latency: " + str(end_time3 - start_time3))
    print(output)

    # If model accurately marked with the correct, expected answer
    # proving the models knowledge
    if expected_answers[i] in output:
      model_acc_count+=1

    print("---------------------------")
  
  print("System: The quiz is finally over, now that took a long time didn't it!\n")
  # Want this to be very close to 15/15
  print(f"System: The models accuracy is {model_acc_count}/15\n")
  file.write(str(datetime.datetime.now()) + ": The models accuracy is: " + str(model_acc_count) + "/15\n")

  if model_acc_count >= 13:
    print("System: The model was highly accurate, and capable of grading a student.\n")
    file.write(str(datetime.datetime.now()) + ": The model was highly accurate.\n")
  else:
    print("System: The model is not accurate enough to successfully grade a student.\n")
    file.write(str(datetime.datetime.now()) + ": The model is not able to accurately grade a student.\n")
  
  
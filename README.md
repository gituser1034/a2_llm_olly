# a2_llm_olly
Olly Love
Ollama + Python LLM Astronomy Study Buddy
Demo Video: https://youtu.be/KCmun3s91HQ 

Overview: An Ai-engineering application using a local Ollama model with Langchain and Chroma DB. 
RAG from a small pdf set generates a True/False quiz to help me study for my upcoming Astronomy exam.
RAG done once at the start and information loaded into local cache lists to speed up.
Model consistently refers to local caches to grade a students answer.

How to run:
Install Ollama on Windows + pull model qwen3:4b
```
create venv (python -m venv venv)
Run venv: venv\Scripts\activate
```

Install these in the active venv:
```
pip install ollama 
pip install langchain langchain-ollama langchain-chroma
ollama pull mxbai-embed-large
pip install pandas
pip install langchain-community
pip install pypdf
pip install langchain-text-splitters
```

Run (Note: I already ran the vector store, no need to rerun that file):
```
To embed: python vectorpdf.py
To run: python main.py
To run input tests: python test_main.py
```

Testing Note:
It is impossible to predetermine what questions the llm will ask, so I 
took questions from one run, and to test inputs I created a seperate document not
generating the questions through RAG, to seed the generated questions. RAG is used for answer generation
only in that file.



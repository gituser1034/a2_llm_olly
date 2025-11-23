# a2_llm_olly
Olly Love
Ollama + Python LLM Study Buddy

How to run:
Install Ollama on Windows + pull model qwen3:4b, create venv (python -m venv venv)
Run venv: venv\Scripts\activate

Install these in the active venv:
pip install ollama 
pip install langchain langchain-ollama langchain-chroma
ollama pull mxbai-embed-large
pip install pandas
pip install langchain-community
pip install pypdf

Run - can change to only using main later:
To embed: python vector.py
To run post-embedding: python main.py



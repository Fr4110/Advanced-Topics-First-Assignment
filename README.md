# Advanced Topics: Text-to-SQL vs Direct Table QA

This repository contains the evaluation infrastructure to compare the Text-to-SQL and Direct Table QA paradigms using open-source LLMs (Llama 3 8B), as described in the accompanying report.

## Requirements and Setup

This project was developed using **Python 3.10+**.

### 1. Install Dependencies
Only the OpenAI Python SDK is required to interface with the local LLM server. Run the following command:
pip install -r requirements.txt

### 2. Local LLM Setup (Ollama)
The evaluation strictly runs completely offline to ensure reproducibility without commercial APIs. You must install Ollama (https://ollama.com/) and pull the specific model used in this evaluation:
ollama pull llama3

Ensure the Ollama server is running locally on port 11434 before starting the script.

### 3. Dataset Configuration
The script evaluates a curated subset of the Spider benchmark. You must place the Spider data in a "data" folder inside the project root, maintaining the following structure:

```text
project_root/
├── data/
│   ├── dev.json
│   └── database/
│       ├── concert_singer/
│       │   └── concert_singer.sqlite
│       └── pets_1/
│           └── pets_1.sqlite
├── run_evaluation.py
└── requirements.txt
```

## Running the Evaluation
To start the pipeline and generate the evaluation metrics, execute:
python run_evaluation.py

The script will output the progress in the console and, upon completion, save all predictions, raw outputs, and aggregated Qatch metrics in the "evaluation_log.json" file.

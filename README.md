# 🤖 Local AI Auditor & Ragas Dashboard

A powerful, privacy-focused Streamlit application designed for **Prompt Engineering**, **Email Output Analysis**, and **RAG (Retrieval-Augmented Generation) Evaluation**. This tool runs entirely on your local machine using **Ollama**, ensuring your data never leaves your computer.

## 🌟 Key Features

- **Prompt Engineering Lab**: Audit and optimize system prompts. Get an "Audit Score," identify flaws, and receive an improved version.
- **Email Output Analyzer**: Specifically designed for teams working with AI-generated email templates. It audits the narrative flow, tone consistency, and logic of JSON-formatted email outputs.
- **RAG Evaluation (Metrics)**: Locally calculate Ragas metrics (**Faithfulness** and **Answer Relevancy**) using your own hardware. No OpenAI API key required.
- **Local-First**: Powered by Ollama. Supports models like Llama 3, Mistral, and Phi-3.

---

## 🚀 Getting Started

### 1. Prerequisites

- **macOS** (Optimized for M1/M2/M3 chips) or Linux/Windows.
- **Python 3.9+**
- **Ollama**: Download and install from [ollama.com](https://ollama.com/).

### 2. Install Local Models

Before running the app, pull the necessary models via your terminal:

```bash
ollama pull llama3
```

### 3. Instalation

Clone this repository and install the required Python libraries:

```bash
# Clone the repo
git clone https://github.com/lukasz1pr/ragas-local-analyzer.git
cd ragas-local-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 4. Running the App

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

## 📊 How to Use the RAG Evaluator

To test the RAG Evaluation tab, upload a CSV file with the following columns:

question: The user query.

contexts: A list of reference documents (formatted as ["text"]).

answer: The generated response from your RAG system.

ground_truth: The ideal/correct answer.

The app will use Local Embeddings and Local LLMs to calculate scores without any external API calls.

## 🛠️ Built With

Streamlit - The UI framework.

Ragas - Evaluation framework for RAG pipelines.

Ollama - Local LLM runtime.

LangChain - Orchestration between LLMs and tools.

## 🔐 Privacy & Security

This project is designed for enterprise-grade privacy. Since it uses OllamaEmbeddings and ChatOllama, all data processing happens locally on your hardware. No data is sent to OpenAI, Google, or any other cloud provider.

Created by Łukasz Prystupa

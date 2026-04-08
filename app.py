import streamlit as st
import pandas as pd
import json
import ast
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 1. Page Configuration
st.set_page_config(page_title="Local AI Auditor", layout="wide", page_icon="🤖")

st.title("🤖 Local AI Auditor & Ragas Dashboard")
st.markdown("---")

# 2. Sidebar Settings
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Select Local Model:",
    ["llama3", "mistral", "phi3"],
    help="Ensure you ran 'ollama pull [model_name]' in terminal."
)

task = st.sidebar.radio(
    "Select Function:",
    ["Prompt Engineering Lab", "Email Output Analyzer", "RAG Evaluation (Metrics)", "System Status"]
)

# Initialize Local LLM via Ollama
llm = ChatOllama(model=selected_model)

# --- SECTION 1: PROMPT ENGINEERING LAB ---
if task == "Prompt Engineering Lab":
    st.header("📝 Prompt Auditor & Optimizer")
    st.info("Critique and improve your system prompts locally.")
    
    user_prompt = st.text_area("Input your System Prompt here:", height=200)
    
    if st.button("Analyze & Audit"):
        if not user_prompt:
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner(f"Running audit using {selected_model}..."):
                try:
                    audit_instruction = (
                        "Analyze the following prompt. Provide: 1. Audit Score (0/10), "
                        "2. Identified Flaws, 3. Improved Version, 4. Brief Explanation."
                    )
                    response = llm.invoke(f"{audit_instruction}\n\nPROMPT:\n{user_prompt}")
                    st.success("Analysis Complete!")
                    st.markdown("### 📊 Audit Report")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Error: {e}")

# --- SECTION 2: EMAIL OUTPUT ANALYZER (JSON) ---
elif task == "Email Output Analyzer":
    st.header("📧 JSON Email Output Auditor")
    st.info("Paste your generated JSON to analyze narrative flow and component synergy.")
    
    json_input = st.text_area("Paste JSON here:", height=300, placeholder='{"generatedFields": {...}}')
    
    if st.button("Audit JSON Content"):
        if not json_input:
            st.warning("Please paste the JSON first.")
        else:
            with st.spinner("Auditing narrative flow..."):
                try:
                    data = json.loads(json_input)
                    fields = data.get("generatedFields", {})
                    formatted_content = "\n".join([f"{k.upper()}: {v}" for k, v in fields.items()])

                    audit_instruction = (
                        "You are a Conversion Copywriting Expert. Analyze these email components "
                        "as a single cohesive message. Ignore placeholder names. "
                        "Focus on: 1. Narrative Arc (Flow between fields), 2. Tone Consistency, "
                        "3. Value Clarity, 4. AI Cliches, 5. Final Rating & Pro-Tip."
                    )
                    
                    response = llm.invoke(f"{audit_instruction}\n\nCOMPONENTS:\n{formatted_content}")
                    st.success("Audit Complete!")
                    st.markdown("### 🎯 Strategic Feedback")
                    st.write(response.content)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Check your brackets and quotes.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- SECTION 3: RAG EVALUATION (METRICS) ---
elif task == "RAG Evaluation (Metrics)":
    st.header("📊 RAG Performance Metrics")
    st.write("Upload a CSV to calculate Faithfulness and Relevancy using Ragas.")
    
    uploaded_file = st.file_uploader("Upload CSV/JSON (Columns: question, contexts, answer, ground_truth)", type=["csv", "json"])
    
    if uploaded_file:
        # Load file with encoding fallbacks
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python')
        
        # Helper function to convert strings like '["text"]' to real Python lists
        def parse_contexts(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return [x]
            return x

        # Apply fix to contexts column if it exists
        if 'contexts' in df.columns:
            df['contexts'] = df['contexts'].apply(parse_contexts)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(3))
        
        if st.button("Run Ragas Evaluation"):
            with st.spinner("Calculating metrics locally (LLM + Embeddings)..."):
                try:
                    # 1. Initialize local embeddings
                    embeddings = OllamaEmbeddings(model=selected_model)
                    
                    # 2. Map pandas to Ragas Dataset
                    dataset = Dataset.from_pandas(df)
                    
                    # 3. Run Evaluation with local LLM AND local Embeddings
                    result = evaluate(
                        dataset, 
                        metrics=[faithfulness, answer_relevancy], 
                        llm=llm,
                        embeddings=embeddings  # <--- TO ROZWIĄZUJE TWÓJ BŁĄD
                    )
                    
                    st.success("Evaluation Finished!")
                    st.markdown("### 📈 Final Results")
                    st.dataframe(result.to_pandas())
                except Exception as e:
                    st.error(f"Ragas Evaluation Error: {e}")

# --- SECTION 4: SYSTEM STATUS ---
else:
    st.header("⚙️ System Status")
    st.write(f"**Connected to Local Engine:** Ollama")
    st.write(f"**Endpoint:** `http://localhost:11434`")
    st.write(f"**Active Model:** `{selected_model}`")
    st.markdown("""
    **Quick Fixes:**
    1. If the model is slow, try a smaller model like `phi3`.
    2. If you get a 'Connection Error', make sure the Ollama app is running in your Menu Bar.
    3. Ensure your CSV has the exact column names: `question`, `contexts`, `answer`, `ground_truth`.
    """)
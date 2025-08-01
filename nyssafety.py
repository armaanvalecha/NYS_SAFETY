import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
import requests
import json

# --- Paths ---
pdf_folder = r"C:\Users\Hp\Desktop\ACCIDENTS"
csv_path = os.path.join(pdf_folder, "extracted.csv")

# --- Extract details ---
def extract_case_details(text):
    text = text.replace('\n', ' ').replace('  ', ' ')
    title = re.search(r'Title:\s*(.*?)\.', text)
    location = re.search(r'Location:\s*(.*?)\.', text)
    date = re.search(r'Dt\.:\s*(\d{2}/\d{2}/\d{4})', text)
    return {
        'title': title.group(1) if title else '',
        'location': location.group(1) if location else '',
        'date': date.group(1) if date else '',
        'text': text
    }

# --- Load data ---
if not os.path.exists(csv_path):
    rows = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
                full_text = "".join(page.extract_text() or "" for page in pdf.pages)
                if full_text.strip():
                    rows.append(extract_case_details(full_text))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

df = pd.read_csv(csv_path).fillna("N/A")
texts = df['text'].tolist()

# --- Embedding ---
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

# --- FAISS Index ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --- Risk Keywords ---
risk_weights = {
    "explosion": 10, "fire": 8, "corrosion": 5, "valve failure": 6,
    "LPG": 9, "oxygen deficiency": 7, "manual override": 5,
    "alarm bypass": 7, "leak": 6, "startup": 4, "shutdown": 3
}

def extract_severity(text):
    match = re.search(r'Loss.*?:.*?([0-9]+\.*[0-9]*)', text)
    return float(match.group(1)) if match else 5.0

# --- LLM answer ---
def generate_answer(question, context):
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question:
{question}

Answer:"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": prompt, "stream": True},
            stream=True
        )
        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                output += data.get("response", "")
        return output
    except Exception as e:
        return f"❌ LLM connection failed: {e}"

# --- Gradio chatbot function ---
def chatbot(query):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, I = index.search(q_emb, k=3)
    similar_texts = [texts[i] for i in I[0]]
    context = "\n\n".join(similar_texts)

    # Chatbot answer
    answer = generate_answer(query, context)

    # Risk score
    keyword_hits = [k for k in risk_weights if k in query.lower()]
    param_weight = sum(risk_weights[k] for k in keyword_hits)
    param_score = min(param_weight / 3, 10)

    severities = [extract_severity(t) for t in similar_texts]
    hist_avg = sum(severities) / len(severities)
    total_score = round(0.4 * param_score + 0.6 * hist_avg, 2)

    sim_accidents = "\n\n".join(f"**{df.iloc[i]['title']}** ({df.iloc[i]['date']})\n{texts[i][:500]}..." for i in I[0])

    return answer, f"{total_score}/10", ", ".join(keyword_hits) or "None", ", ".join(f"{s:.1f}" for s in severities), sim_accidents

# --- Gradio UI ---
demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="🔍 Describe a situation (e.g., LPG leak in compressor)", lines=2, placeholder="Enter accident scenario..."),
    outputs=[
        gr.Textbox(label="🧠 Answer"),
        gr.Textbox(label="🔥 Risk Score"),
        gr.Textbox(label="📌 Detected Parameters"),
        gr.Textbox(label="📊 Past Accident Severities"),
        gr.Textbox(label="📚 Top 3 Similar Accidents")
    ],
    title="Accident Case Chatbot + Risk Analyzer",
    description="Ask safety-related questions or describe accident scenarios to get answers, risk scores, and similar incidents."
)

demo.launch(share=True)

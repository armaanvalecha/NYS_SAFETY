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

"""
Accident Case Chatbot + Risk Analyzer (TinyLLaMA / Ollama backend)
---------------------------------------------------------------
*   Extracts accident cases from PDFs the first time the app runs, then
    caches them in `extracted.csv`.
*   Uses MiniLM embeddings + FAISS to retrieve the 3 most‑similar cases.
*   Calls TinyLLaMA through the Ollama REST API to answer the user’s
    query **with context**.
*   Calculates a **0‑10 risk score** that combines:
        • keyword‑based parameter score (0‑10)
        • historical severity average (0‑10)
    The final score is clipped so it can never exceed 10.
"""

# ─────────────────────────────── Paths ──────────────────────────────
pdf_folder = r"C:\Users\Hp\Desktop\ACCIDENTS"
csv_path   = os.path.join(pdf_folder, "extracted.csv")

# ──────────────────────── PDF → structured rows ─────────────────────

def extract_case_details(text: str) -> dict:
    """Return a dict with title, location, date, full text."""
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return {
        "title":    re.search(r"Title:\s*(.*?)\.", text).group(1)     if re.search(r"Title:\s*(.*?)\.", text)     else "",
        "location": re.search(r"Location:\s*(.*?)\.", text).group(1)  if re.search(r"Location:\s*(.*?)\.", text)  else "",
        "date":     re.search(r"Dt\.?:\s*(\d{2}/\d{2}/\d{4})", text).group(1) if re.search(r"Dt\.?:\s*(\d{2}/\d{2}/\d{4})", text) else "",
        "text":     text.strip()
    }

if not os.path.exists(csv_path):
    rows = []
    for fn in os.listdir(pdf_folder):
        if fn.lower().endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, fn)) as pdf:
                txt = "".join(page.extract_text() or "" for page in pdf.pages)
            if txt.strip():
                rows.append(extract_case_details(txt))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

# ───────────────────────────── Embeddings ───────────────────────────
df     = pd.read_csv(csv_path).fillna("N/A")
texts  = df["text"].tolist()
model  = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeds = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")

index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)

# ─────────────────────────── Risk logic ─────────────────────────────
RISK_WEIGHTS = {
    "explosion": 10, "fire": 8, "corrosion": 5, "valve failure": 6,
    "lpg": 9, "oxygen deficiency": 7, "manual override": 5,
    "alarm bypass": 7, "leak": 6, "startup": 4, "shutdown": 3
}

SEVERITY_PATTERN = re.compile(r"Loss[^:]*?:[^0-9]*?([0-9]+\.?[0-9]*)", re.I)

def param_score(query: str) -> float:
    """Keyword‑weighted parameter risk (0‑10)."""
    hits = [k for k in RISK_WEIGHTS if k in query.lower()]
    score = min(sum(RISK_WEIGHTS[k] for k in hits) / 3, 10)  # scale
    return score, hits

def hist_severity(texts: list[str]) -> tuple[float, list[float]]:
    """Average historical severity scaled 0‑10."""
    severities = []
    for t in texts:
        m = SEVERITY_PATTERN.search(t)
        if m:
            val = float(m.group(1))
            # crude scaling: assume raw 0‑100 maps to 0‑10
            val = min(val / 10 if val > 10 else val, 10)
        else:
            val = 5.0  # default medium severity
        severities.append(val)
    avg = sum(severities) / len(severities)
    return avg, severities

# ─────────────────────── Ollama / TinyLLaMA call ────────────────────
API_URL = "http://localhost:11434/api/generate"

def llm_answer(question: str, context: str) -> str:
    prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    try:
        r = requests.post(API_URL, json={"model": "tinyllama", "prompt": prompt, "stream": True}, stream=True)
        out = ""
        for line in r.iter_lines():
            if line:
                out += json.loads(line.decode())["response"]
        return out.strip()
    except Exception as e:
        return f"❌ LLM error: {e}"

# ───────────────────────── Gradio callback ──────────────────────────

def chatbot(query: str):
    # similarity search
    q_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    _, idx = index.search(q_vec, k=3)
    ctx_texts = [texts[i] for i in idx[0]]
    context   = "\n\n".join(ctx_texts)

    # LLM answer
    answer = llm_answer(query, context)

    # risk scores
    p_score, hits = param_score(query)
    h_avg, sev_list = hist_severity(ctx_texts)

    total = round(min(0.4 * p_score + 0.6 * h_avg, 10), 2)  # ALWAYS 0‑10

    sim_accidents = "\n\n".join(
        f"**{df.iloc[i]['title']}** ({df.iloc[i]['date']})\n{texts[i][:400]}..." for i in idx[0]
    )

    return (
        answer,
        f"{total}/10",
        ", ".join(hits) or "None",
        ", ".join(f"{s:.1f}" for s in sev_list),
        sim_accidents,
    )

# ───────────────────────────── Gradio UI ────────────────────────────
demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="🔍 Describe a situation", lines=2, placeholder="Enter accident scenario..."),
    outputs=[
        gr.Textbox(label="🧠 Answer"),
        gr.Textbox(label="🔥 Risk Score"),
        gr.Textbox(label="📌 Detected Parameters"),
        gr.Textbox(label="📊 Past Accident Severities"),
        gr.Textbox(label="📚 Top 3 Similar Accidents"),
    ],
    title="Accident Case Chatbot + Risk Analyzer (TinyLLaMA)",
    description="Ask safety questions or describe incidents to get answers, risk scores (0‑10), and similar historical accidents.",
)

demo.launch(share=True)

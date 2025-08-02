# EchoSketch — Multimodal LLM Chatbot

**EchoSketch** is a Gradio-based chatbot UI powered by a Tornado + GPT‑4o backend. It supports:
- ⚡ Text-based RAG over a local corpus (e.g. `cat-facts.txt`)
- 📷 Image upload and description via GPT‑4o + embeddings
- 🎯 Retrieval-augmented generation with citation-style context

---

## 🚀 Features

- **Text & Image Retrieval‑Augmented Chat**  
  Ask questions grounded in your dataset and optionally uploaded images.
- **Multimodal Support**  
  Upload an image to get an auto-generated description and add it to the RAG retrieval pipeline.
- **Citation Style Transparency**  
  Responses include inline citation metadata linking back to your files or uploaded images.
- **Lightweight & Local**  
  Runs fully locally with Tornado backend and Gradio frontend—no cloud required.
- **Secure API Key Handling**  
  Stores your OpenAI key in `.env`, not in source code.

---

## 📁 Repository Structure

```
EchoSketch‑Multimodal‑LLM‑Chatbot/
├── app.py               # Tornado backend server with ask + image endpoints
├── cat‑facts.txt        # Sample text dataset used for RAG retrieval
├── system_prompt.txt    # Prompt template with {context}
├── frontend.py          # Gradio UI for chat and image upload
├── requirements.txt     # Required third‑party packages
├── .gitignore           # Excludes .env, __pycache__, etc.
└── README.md            # This file
```

---

## ⚙️ Setup & Usage

### ✅ Prerequisites
- Python **3.10**
- OpenAI API key
- Optional: `cat-facts.txt` dataset

### 1. Clone the repo

```bash
git clone https://github.com/anjorisarabhai/EchoSketch-Multimodal-LLM-Chatbot.git
cd EchoSketch-Multimodal-LLM-Chatbot
```

### 2. Create `.env` file

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=sk-...
```

### 3. Install dependencies

Install all necessary packages using:

```powershell
py -3.10 -m pip install -r requirements.txt
```

### 4. Run the backend

```powershell
py -3.10 app.py
```

You should see:

```
Server running at http://localhost:8888
```

### 5. Run the frontend (in a separate terminal)

```powershell
py -3.10 frontend.py
```

This launches a Gradio UI (usually at `http://127.0.0.1:7860`).

---

## 🧪 How It Works

1. **Data Ingestion**  
   `app.py` loads local text data (`cat-facts.txt`), builds embeddings, and stores them in memory.
2. **Image Upload**  
   Uploading an image sends it to `/upload_image`, triggers GPT‑4o description, converts that description to an embedding, and adds it to the image vector DB.
3. **Question Answering**  
   Sending a question to `/ask` retrieves relevant text chunks + image descriptions using cosine similarity, constructs a system prompt, then generates a response via GPT‑4o with context.
4. **Gradio UI**  
   `frontend.py` provides:
   - An image upload button
   - A chat interface
   - A typing animation
   - Formatted citations pulled from your response JSON

---

## 💾 Install Dependencies

Install everything listed in [`requirements.txt`](./requirements.txt):

```bash
py -3.10 -m pip install -r requirements.txt
```

---

## 📝 Git Ignore

Sensitive and unwanted files are already excluded via [` .gitignore`](./.gitignore), including:
- `.env`
- `__pycache__/`
- Compiled Python files (`*.pyc`)

---

## 📌 Notes & Tips

- To avoid `404 Not Found` errors in your browser, visiting `http://localhost:8888/` will show a simple home page if you add a `MainHandler` in `app.py`.
- The RAG threshold is set at relevance score ≥ 0.2. You can **tune this** or change the number of retrieved contexts in `retrieve_combined_context()`.
- For citation support, your `ask()` function must return both `response` and `citations` arrays, matching your Gradio `format_citations()` logic.

---

## 🧠 Credits & Inspiration

This project is inspired by modern multimodal RAG chatbots with Gradio frontends, Tornado backends, and GPT‑4o multimodal capabilities.

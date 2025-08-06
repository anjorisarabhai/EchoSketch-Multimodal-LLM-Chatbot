import tornado.web
import tornado.ioloop
import tornado.httpserver
from openai import AsyncOpenAI
import asyncio
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import docx
from pathlib import Path
from email import policy
from email.parser import BytesParser
import re

# --- LOAD API KEY FROM ENV ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# --- TEXT VECTOR DB ---
TEXT_VECTOR_DB = []

# --- IMAGE DESCRIPTIONS DB ---
IMAGE_VECTOR_DB = []

# --- COSINE SIMILARITY ---
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b)

# --- EMBEDDING FUNCTION ---
async def get_embedding(text: str):
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

# --- PARSE PDF ---


def parse_pdf(path):
    doc = fitz.open(path)
    chunks = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if not text:
            continue

        # Split by "Clause X:" or "Section X:"
        sections = re.split(r"(Clause\s+\d+:|Section\s+\d+:)", text)
        
        # Recombine titles + content
        i = 0
        while i < len(sections) - 1:
            if re.match(r"(Clause\s+\d+:|Section\s+\d+:)", sections[i]):
                section_title = sections[i].strip()
                content = sections[i + 1].strip()
                if content:
                    chunks.append({
                        "chunk": f"{section_title} {content}",
                        "document_name": path.name,
                        "page_number": page_num,
                        "section_title": section_title
                    })
                i += 2
            else:
                i += 1

    return chunks

# --- PARSE DOCX ---
def parse_docx(path):
    doc = docx.Document(path)
    chunks = []

    current_section = "Unknown Section"
    current_text = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect section headers
        if re.match(r"(Section\s+\d+:?|Clause\s+\d+:?|Confidentiality|Governing Law|Term|Obligations)", text, re.IGNORECASE):
            # If we had previous section content, store it
            if current_text:
                chunks.append({
                    "chunk": "\n".join(current_text),
                    "document_name": path.name,
                    "section_title": current_section
                })
                current_text = []

            current_section = text  # New heading
        else:
            current_text.append(text)

    # Append last section
    if current_text:
        chunks.append({
            "chunk": "\n".join(current_text),
            "document_name": path.name,
            "section_title": current_section
        })

    return chunks

# --- PARSE TXT ---

def parse_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_section = "Unknown Section"
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headings like: Section 1, Clause 3, Confidentiality
        if re.match(r"(Section\s+\d+:?|Clause\s+\d+:?|Confidentiality|Governing Law|Term|Obligations)", line, re.IGNORECASE):
            if current_text:
                chunks.append({
                    "chunk": "\n".join(current_text),
                    "document_name": path.name,
                    "section_title": current_section
                })
                current_text = []

            current_section = line
        else:
            current_text.append(line)

    if current_text:
        chunks.append({
            "chunk": "\n".join(current_text),
            "document_name": path.name,
            "section_title": current_section
        })

    return chunks

# --- PARSE EML ---
def parse_eml(path):
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    email_from = msg["From"]
    email_to = msg["To"]
    subject = msg["Subject"]
    body = msg.get_body(preferencelist=('plain'))

    if body:
        content = body.get_content().strip()
        return [{
            "chunk": content,
            "document_name": path.name,
            "section_title": f"Subject: {subject or 'No Subject'}",
            "from": email_from,
            "to": email_to
        }]
    return []

# --- BUILD TEXT VECTOR DB ---
async def build_text_vector_db():
    print("📄 Building text vector DB from documents/ ...")
    all_chunks = []

    docs_path = Path("documents")
    for file in docs_path.glob("*"):
        if file.suffix.lower() == ".pdf":
            all_chunks.extend(parse_pdf(file))
        elif file.suffix.lower() == ".docx":
            all_chunks.extend(parse_docx(file))
        elif file.suffix.lower() == ".txt":
            all_chunks.extend(parse_txt(file))
        elif file.suffix.lower() == ".eml":
            all_chunks.extend(parse_eml(file))
        else:
            print(f"⚠️ Skipping unsupported file: {file.name}")

    print(f"Total chunks found: {len(all_chunks)}")
    tasks = [get_embedding(chunk["chunk"]) for chunk in all_chunks]
    embeddings = await asyncio.gather(*tasks)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb
        TEXT_VECTOR_DB.append(chunk)

    print("✅ Text vector DB ready.")

# --- IMAGE TO TEXT VIA GPT-4o ---
async def describe_image_base64(b64: str, mime: str):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                ]}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error describing image:", e)
        raise

# --- SYSTEM PROMPT ---
def load_system_prompt(context: str) -> str:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{context}", context)

# --- RETRIEVAL ---
async def retrieve_combined_context(query: str, top_n=3):
    query_emb = await get_embedding(query)

    text_scores = [(item, cosine_similarity(query_emb, item["embedding"])) for item in TEXT_VECTOR_DB]
    image_scores = [(item, cosine_similarity(query_emb, item["embedding"])) for item in IMAGE_VECTOR_DB]

    text_scores.sort(key=lambda x: x[1], reverse=True)
    image_scores.sort(key=lambda x: x[1], reverse=True)

    combined = text_scores[:top_n] + image_scores[:top_n]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_n * 2]

# --- RAG QA FUNCTION with CITATIONS ---
async def ask(query: str):
    results = await retrieve_combined_context(query)
    if not results or max(score for _, score in results) < 0.2:
        return "I don't have information about that based on the current data."

    context_parts = []
    citations = []

    for item, _ in results:
        chunk = item["chunk"]
        if "document_name" in item:
            if "page_number" in item:
                citation = f"(source: {item['document_name']}, page {item['page_number']})"
            elif "section_title" in item:
                citation = f"(source: {item['document_name']}, section {item['section_title']})"
            else:
                citation = f"(source: {item['document_name']})"
        elif "image_name" in item:
            citation = f"(image: {item['image_name']}, uploaded at {item['uploaded_at']})"
        else:
            citation = "(source: unknown)"

        context_parts.append(f"- {chunk} {citation}")
        citations.append(citation)

    full_context = "\n".join(context_parts)
    system_prompt = load_system_prompt(full_context)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- TORNADO HANDLERS ---
class AskHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            question = data.get("question", "")
            if not question:
                self.set_status(400)
                return self.write({"error": "Missing 'question' field"})
            response = await ask(question)
            self.write({"response": response})
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

class UploadImageHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            fileinfo = self.request.files["image"][0]
            mime = fileinfo.get("content_type", "image/jpeg")
            b64 = base64.b64encode(fileinfo["body"]).decode("utf-8")

            description = await describe_image_base64(b64, mime)
            embedding = await get_embedding(description)

            IMAGE_VECTOR_DB.append({
                "chunk": description,
                "embedding": embedding,
                "image_name": fileinfo["filename"],
                "uploaded_at": datetime.now().isoformat()
            })

            self.write({"description": description})
        except Exception as e:
            print("Upload Error:", str(e))
            self.set_status(500)
            self.write({"error": str(e)})

class SlotExtractionHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            query = data.get("query", "").strip()
            if not query:
                self.set_status(400)
                return self.write({"error": "Missing 'query' field"})

            prompt = f"""
Extract structured information from the query below.

Query: "{query}"

Return JSON with the following fields: age, gender, procedure, location, policy_age (in months).
"""
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            # Try parsing to JSON
            try:
                structured_data = json.loads(content)
                self.write({"slots": structured_data})
            except json.JSONDecodeError:
                self.write({
                    "slots": {},
                    "raw_response": content,
                    "warning": "Response could not be parsed as JSON. Returned raw text instead."
                })

        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})


# --- SETUP ---
def make_app():
    return tornado.web.Application([
        (r"/ask", AskHandler),
        (r"/upload_image", UploadImageHandler),
        (r"/extract_slots", SlotExtractionHandler),
    ])

async def main():
    await build_text_vector_db()
    app = make_app()
    app.listen(8888)
    print("🚀 Server running at http://localhost:8888")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())

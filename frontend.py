import gradio as gr
import requests
import time

BACKEND_URL = "http://localhost:8888"

# ---------- Slot Extraction ----------
def extract_slots(query):
    if not query.strip():
        return "Please enter a query."

    try:
        res = requests.post(f"{BACKEND_URL}/extract_slots", json={"query": query})
        res.raise_for_status()
        data = res.json()
        if "slots" in data and data["slots"]:
            return f"**Extracted Slots:**\n\n```json\n{data['slots']}\n```"
        elif "raw_response" in data:
            return f"Raw response:\n\n{data['raw_response']}"
        else:
            return "No slots found."
    except Exception as e:
        return f"Error extracting slots:\n{str(e)}"

# ---------- Image Upload ----------
def upload_image(image):
    if image is None:
        return "Please upload an image first."

    files = {'image': ('image.jpg', open(image, "rb"), 'image/jpeg')}
    try:
        response = requests.post(f"{BACKEND_URL}/upload_image", files=files)
        response.raise_for_status()
        desc = response.json().get("description", "")
        return f"Image analyzed successfully!\n\n**Description**: {desc}"
    except Exception as e:
        return f"Error uploading image:\n{str(e)}"

# ---------- Citation Formatting ----------
def format_citations(citations):
    if not citations:
        return ""
    formatted = "\n\n**Citations:**\n"
    for c in citations:
        if c.get("source") == "image":
            formatted += f"- Image: *{c.get('image_name')}* — \"{c.get('description')}\"\n"
        else:
            formatted += f"- File: *{c.get('source')}*, Line {c.get('line_number')}: \"{c.get('content')}\"\n"
    return formatted

# ---------- Chat QA ----------
def ask_question(question, history):
    if not question.strip():
        return history, "", gr.update(visible=False)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "..."})  # placeholder

    yield history, "", gr.update(visible=True)

    try:
        res = requests.post(f"{BACKEND_URL}/ask", json={"question": question})
        res.raise_for_status()
        data = res.json()
        answer = data.get("response", "No response.")
        citations = data.get("citations", [])
        answer += format_citations(citations)
    except Exception as e:
        answer = f"Error: {str(e)}"

    # Typing animation
    typed = ""
    for char in answer:
        typed += char
        history[-1]["content"] = typed
        yield history, "", gr.update(visible=True)
        time.sleep(0.01)

    yield history, "", gr.update(visible=False)

# ---------- UI ----------
with gr.Blocks(css="""
    .gr-chat-message.user {
        text-align: right;
        background-color: #d4f4ff;
        padding: 8px;
        border-radius: 10px;
        margin-left: 40%;
    }
    .gr-chat-message.assistant {
        text-align: left;
        background-color: #f0f0f0;
        padding: 8px;
        border-radius: 10px;
        margin-right: 40%;
    }
""") as demo:
    gr.Markdown("## 🧠 Multimodal RAG Chatbot (Text + Image + Slot Extraction)")

    # --- Image Upload ---
    with gr.Row():
        image_input = gr.Image(label="Upload an Image", type="filepath", height=200)
        upload_button = gr.Button("Upload & Analyze Image")
    image_status = gr.Markdown()
    upload_button.click(fn=upload_image, inputs=image_input, outputs=image_status)

    # --- Chat QA ---
    chatbot = gr.Chatbot(label="Chat", type="messages")
    with gr.Row(equal_height=True):
        question = gr.Textbox(
            placeholder="Ask anything related to the text or uploaded image...",
            lines=1,
            show_label=False
        )
        send_btn = gr.Button("Send")

    loading = gr.Text("Thinking...", visible=False)

    question.submit(
        fn=ask_question,
        inputs=[question, chatbot],
        outputs=[chatbot, question, loading],
        show_progress=True
    )
    send_btn.click(
        fn=ask_question,
        inputs=[question, chatbot],
        outputs=[chatbot, question, loading],
        show_progress=True
    )

    # --- Slot Extraction ---
    gr.Markdown("### 🧾 Slot Extraction from Query")
    slot_input = gr.Textbox(placeholder="Enter query to extract structured slots...")
    slot_output = gr.Markdown()
    slot_button = gr.Button("Extract Slots")

    slot_button.click(fn=extract_slots, inputs=slot_input, outputs=slot_output)

demo.launch()

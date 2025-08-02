import gradio as gr
import requests
import time

BACKEND_URL = "http://localhost:8888"

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

    # Simulate typing effect
    typed = ""
    for char in answer:
        typed += char
        history[-1]["content"] = typed
        yield history, "", gr.update(visible=True)
        time.sleep(0.01)

    yield history, "", gr.update(visible=False)

# Gradio UI
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
    gr.Markdown("## RAG Chatbot – Text + Image Understanding")

    with gr.Row():
        image_input = gr.Image(label="Upload an Image", type="filepath", height=200)
        upload_button = gr.Button("Upload & Analyze Image")

    image_status = gr.Markdown("")

    upload_button.click(fn=upload_image, inputs=image_input, outputs=image_status)

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

demo.launch()

from ollama import chat, pull, ResponseError
import gradio as gr

# Specify the model name
model_name = "deepseek-r1:1.5b"  # Replace with the model name you want to use

# Ensure the model is available locally
try:
    pull(model_name)
except ResponseError as e:
    if e.status_code == 404:
        print(f"Model {model_name} not found. Please ensure it is pulled using 'ollama pull {model_name}'.")

# Chatbot function
def ollama_chatbot(user_message, chat_history):
    try:
        # Format the chat history
        messages = [{"role": "user", "content": msg[0]} for msg in chat_history]
        messages.append({"role": "user", "content": user_message})
        
        # Stream the response from the Ollama model
        stream = chat(model=model_name, messages=messages, stream=True)
        bot_response = ""
        for chunk in stream:
            bot_response += chunk['message']['content']
        
        # Append the user's input and bot's response to the chat history
        chat_history.append((user_message, bot_response))
        return chat_history, chat_history
    except ResponseError as e:
        return f"Error: {e.error}", chat_history

# Gradio UI
with gr.Blocks() as ollama_ui:
    gr.Markdown("### 🤖 Chatbot Powered by Ollama")
    
    chatbot = gr.Chatbot(label="Ollama Chatbot")
    user_input = gr.Textbox(label="Your Message", placeholder="Type your question here...")
    submit_button = gr.Button("Send")
    clear_button = gr.Button("Clear Chat")
    
    chat_state = gr.State([])  # To maintain chat history
    
    # Bind the functions to the UI
    submit_button.click(ollama_chatbot, inputs=[user_input, chat_state], outputs=[chatbot, chat_state])
    clear_button.click(lambda: ([], []), inputs=[], outputs=[chatbot, chat_state])

# Launch the app
ollama_ui.launch()

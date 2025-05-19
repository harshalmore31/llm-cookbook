import gradio as gr
from litellm import completion
import os
from dotenv import load_dotenv

load_dotenv()

# Provider and model mappings
providers = {
    'OpenAI': ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    'gemini': ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b","gemini-2.0-flash-exp"],
    'Groq': ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    'Mistral AI': ["mistral-medium", "mistral-large", "mistral-large-2", "mixtral-8x22b"],
    'Anthropic': ["claude-3.5-haiku", "claude-3.5-sonnet", "claude-3-opus"],
}

# Mapping of providers to environment variables
provider_env_map = {
    'OpenAI': 'OPENAI_API_KEY',
    'gemini': 'GEMINI_API_KEY',
    'Groq': 'GROQ_API_KEY',
    'Mistral AI': 'MISTRAL_API_KEY',
    'Anthropic': 'ANTHROPIC_API_KEY'
}

# Gradio theme
theme = gr.themes.Ocean(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="gray",
)

def chat_with_model(provider, model, api_key, message, chat_history):
    """
    Function to interact with the selected model.
    """
    
    # Set the API key in environment variable dynamically
    os.environ[provider_env_map[provider]] = api_key
    
    chat_history = chat_history or []

    try:
        # Construct the message
        messages = []
        for turn in chat_history:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})
        messages.append({"role": "user", "content": message})

        # Make API call
        response = completion(
            model= f"{provider.lower()}/{model}",
            messages=messages,
        )

        # Get the response content
        bot_message = response.choices[0].message.content
        chat_history.append((message,bot_message))
        
        return "", chat_history
    
    except Exception as e:
         chat_history.append((message, f"An error occurred: {e}"))
         return "", chat_history


with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Chat with Different LLM Providers")
    gr.Markdown(" **Warning:** Please be cautious when entering your API keys, especially in public environments. **Do not share your API keys.**")

    with gr.Row():
        with gr.Column():
            provider_dropdown = gr.Dropdown(
                choices=list(providers.keys()),
                label="Select Provider",
            )
            model_dropdown = gr.Dropdown(
                choices=[],
                label="Select Model",
            )
            api_key_input = gr.Textbox(
                label="Enter API Key", type="password",
            )
        
    chatbot = gr.Chatbot(
        label="Chatbot",
    )
    chat_input = gr.Textbox(
        label="Enter Message",
    )
    clear_button = gr.ClearButton([chat_input, chatbot])
        

    provider_dropdown.change(
        lambda provider: gr.Dropdown(choices=providers[provider]),
        inputs=provider_dropdown,
        outputs=model_dropdown,
    )
    
    chat_input.submit(
       chat_with_model,
       [provider_dropdown, model_dropdown, api_key_input, chat_input, chatbot],
       [chat_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
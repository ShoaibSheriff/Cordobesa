import spaces
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly (as in your original code)
tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-13B-v0.1")
model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-13B-v0.1", device_map="auto")

# You must wrap your logic in a function to use the @spaces.GPU decorator
@spaces.GPU
def generate(text):

# We tell the model specifically: "This is the user's command"
    prompt = f"<|im_start|>user\nTranslate this text from Argentinian Spanish to american english. Pay special attention to tone. If there are any jokes, idioms, metaphors in the text, be sure to point it out. Text: {text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Prepare the input text (from your line 9-10)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        do_sample=False  # Better for literal translation
    )

    # --- CHANGE 3: SLICE THE OUTPUT ---
    # We only take the tokens AFTER the prompt_length
    new_tokens = outputs[0][prompt_length:]
    
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Create the Gradio interface (required for ZeroGPU)
demo = gr.Interface(fn=generate, inputs="text", outputs="text")
demo.launch()
import spaces
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly (as in your original code)
tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-13B-v0.1")
model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-13B-v0.1", device_map="auto")

# You must wrap your logic in a function to use the @spaces.GPU decorator
@spaces.GPU
def generate(text):

    prompt = f"Translate the following text into Argentinian Spanish. Ensure regional vocabulary and grammar are correct.\nText: {text}"
    
    # Prepare the input text (from your line 9-10)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the output (from your line 14)
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode and return (from your line 17)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Create the Gradio interface (required for ZeroGPU)
demo = gr.Interface(fn=generate, inputs="text", outputs="text")
demo.launch()
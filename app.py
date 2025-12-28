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
    prompt = f"<|im_start|>user\n 1. Translate this text from Argentinian Spanish to american english. Pay special attention to tone. 2. If there are any jokes, idioms, metaphors in the text, be sure to point it out. Text: {text}<|im_end|>\n<|im_start|>assistant\n"
    
    # Prepare the input text (from your line 9-10)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_length = inputs.input_ids.shape[1]

      # Use a large max_new_tokens ceiling to prevent cutoffs
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # Slice output to remove the input text from the result box
    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Corrected UI Layout
# fill_height=True expands components vertically to window height
# fill_width=True removes side margins for a full-screen feel
with gr.Blocks(fill_height=True, fill_width=True) as demo:
    gr.Markdown("# 🇦🇷 Spanish Localizer")
    
    with gr.Column(scale=1): 
        input_box = gr.Textbox(
            label="Original Text", 
            placeholder="Enter text to translate...", 
            lines=5 
        )
        
        # scale=1 is the key change to make this box grow
        output_box = gr.Textbox(
            label="Argentinian Spanish Translation", 
            interactive=False, 
            lines=15, 
            scale=1 
        )
        
        submit_btn = gr.Button("Translate", variant="primary")

    # Link the button to the function
    submit_btn.click(fn=generate, inputs=input_box, outputs=output_box)

# Final step: launch the demo that was defined in the 'with' block
demo.launch()

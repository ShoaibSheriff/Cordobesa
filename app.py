# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-13B-v0.1")
model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-13B-v0.1")

# 3. Prepare the input text and convert it to model inputs (tensors)
prompt = "How to interact with a model on Hugging Face"
inputs = tokenizer(prompt, return_tensors="pt")

# 4. Generate the output
# For larger models, setting device_map="auto" can help load it onto the fastest device
outputs = model.generate(**inputs, max_new_tokens=50)

# 5. Decode the output tokens back into readable text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)
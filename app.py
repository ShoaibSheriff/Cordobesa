import spaces
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os 
from huggingface_hub import login 
import librosa
from moviepy import VideoFileClip
import tempfile

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

#scribe_model = whisper.load_model("large-v3-turbo")

scribe_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", chunk_length_s=30, torch_dtype=torch.float16, device="cuda")

@spaces.GPU
def generate(text):

# We tell the model specifically: "This is the user's command"

    prompt = f"""<|im_start|>user
Identify and translate the Argentinian 'porteño' nuances in this text.

Text: "{text}"

Please provide the output in this EXACT format:
1. LINGUISTIC ANALYSIS: List any Argentinian idioms (lunfardo), metaphors, or jokes found. Explain their cultural meaning.
2. ENGLISH TRANSLATION: A natural English version that captures the "vibe" (not just literal words).
<|im_end|>
<|im_start|>assistant
"""
    
    # Prepare the input text (from your line 9-10)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    prompt_length = inputs.input_ids.shape[1]

      # Use a large max_new_tokens ceiling to prevent cutoffs
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # Slice output to remove the input text from the result box
    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)



# --- UPDATE YOUR GENERATE FUNCTION ---
# We modify it to handle either raw text or audio
@spaces.GPU
def process_audio(file_path):
    if file_path is None:
        return "", ""

    if file_path.endswith(".mp4"):
        video = VideoFileClip(file_path)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
        audio_path=temp_audio
    else :
        audio_path=file_path
    
    # Scribe logic with Argentinian context prompt
    # The initial_prompt helps Whisper expect 'sh' sounds and slang
    #result = scribe_model.transcribe(
    #    audio_path, 
    #    initial_prompt="Transcripción de una charla argentina con lunfardo y modismos de Buenos Aires."
    #)

    prompt_text = "Transcripción de una charla argentina con lunfardo y modismos de Buenos Aires."
    forced_prompt_ids = scribe_pipe.tokenizer.get_prompt_ids(prompt_text, return_tensors="pt").to("cuda")

    result = scribe_pipe(audio_path, return_timestamps="word", chunk_length_s=30, generate_kwargs={"language":"spanish", "prompt_ids": forced_prompt_ids})
    
    transcription = result["text"]

    
    # Now feed that transcription into your existing Llama logic
    analysis_and_translation = generate(transcription)
    
    return transcription, analysis_and_translation

# Load model directly (as in your original code)
#tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-13B-v0.1")
#model = AutoModelForCausalLM.from_pretrained("Unbabel/TowerInstruct-13B-v0.1", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-LLama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-LLama-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa")


# Corrected UI Layout
# fill_height=True expands components vertically to window height
# fill_width=True removes side margins for a full-screen feel
with gr.Blocks(fill_height=True, fill_width=True) as demo:
    gr.Markdown("# 🇦🇷 Spanish Localizer")
    
    with gr.Column(scale=1): 

        audio_input = gr.File(label="Upload MP3/MP4", file_types=[".mp3", ".mp4", ".wav"])

        # NEW: Show the scribe output on screen
        transcript_box = gr.Textbox(label="Scribe Output (Transcription)", interactive=False, lines=5)
        
        # scale=1 is the key change to make this box grow
        output_box = gr.Textbox(
            label="Argentinian Spanish Translation", 
            interactive=False, 
            lines=15, 
            scale=1 
        )
        
           # NEW: Button for Audio processing
        audio_btn = gr.Button("Transcribe & Analyze Audio", variant="primary")

    # Link the button to the function
    # NEW: Link for audio processing
    audio_btn.click(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[transcript_box, output_box]
    )

# Final step: launch the demo that was defined in the 'with' block
demo.launch()

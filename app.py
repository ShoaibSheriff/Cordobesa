import os
import torch
import gradio as gr
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from moviepy import VideoFileClip, AudioFileClip # Pinned to standard editor for stability
import spaces

# Login
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# 1. Transcription Pipeline (GPU)
scribe_pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v3-turbo", 
    chunk_length_s=30, 
    torch_dtype=torch.float16, 
    device="cuda"
)

# 2. LLM Model Setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-LLama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-LLama-3.1-8B-Instruct", 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="sdpa"
)

def prepare_audio(file_path):
    if not file_path:
        return None
    actual_path = file_path.name if hasattr(file_path, 'name') else file_path
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    # Normalize for Whisper/Diarization requirements
    if actual_path.lower().endswith((".mp4", ".mov", ".mkv")):
        clip = VideoFileClip(actual_path)
    else:
        clip = AudioFileClip(actual_path)
        
    clip.audio.write_audiofile(
        temp_audio, 
        fps=16000, 
        nbytes=2, 
        codec='pcm_s16le', 
        ffmpeg_params=["-ac", "1", "-af", "loudnorm=I=-20:TP=-1.5:LRA=11"],
        logger=None
    )
    clip.close()
    return temp_audio

@spaces.GPU(duration=120)
def generate(text):
    prompt = f"""<|im_start|>user
You are an expert in Argentinian Spanish (Rioplatense). 
Analyze the following SEGMENT of a conversation. Identify 'porteño' nuances, lunfardo, and cultural context.

Text Segment:
"{text}"

Please provide the output in this EXACT format:
1. LINGUISTIC ANALYSIS: List idioms, lunfardo, or metaphors found in this segment.
2. ENGLISH TRANSLATION: A natural English version that captures the "vibe" (not literal).
<|im_end|>
<|im_start|>assistant"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[1]
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1600, 
        do_sample=False,
        temperature=0.0 
    )
    
    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

@spaces.GPU(duration=120)
def scribe_audio(audio_path):
    prompt_text = "Transcripción de una charla argentina con lunfardo y modismos de Buenos Aires."
    forced_prompt_ids = scribe_pipe.tokenizer.get_prompt_ids(prompt_text, return_tensors="pt").to("cuda")

    result = scribe_pipe(
        audio_path, 
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.2, 
            "language":"spanish", 
            "prompt_ids": forced_prompt_ids
        }
    )
    return result

# --- Main Controller (CPU Only) ---
# def process_audio(file_path, chunk_size, overlap):
#     if not file_path:
#         return None, "No file uploaded.", None
        
#     audio_path = prepare_audio(file_path)
    
#     # Run GPU transcription
#     full_transcript = scribe_audio(audio_path)["text"]
    
#     # Dynamic Chunking Logic
#     words = full_transcript.split()
#     chunks = []
    
#     # Guard against infinite loops if overlap is misconfigured
#     step = max(1, int(chunk_size - overlap))
    
#     for i in range(0, len(words), step):
#         chunk = " ".join(words[i : i + int(chunk_size)])
#         chunks.append(chunk)
#         if i + int(chunk_size) >= len(words):
#             break

#     # Run GPU analysis for each chunk
#     final_report = []
#     for i, chunk_text in enumerate(chunks):
#         analysis = generate(chunk_text)
#         final_report.append(f"### PART {i+1} ANALYSIS\n{analysis}")

#     full_analysis = "\n\n".join(final_report)
#     return full_transcript, full_analysis, audio_path

def process_audio(file_path, chunk_size, overlap):
    if not file_path:
        yield "", "No file uploaded.", None
        return
        
    audio_path = prepare_audio(file_path)
    
    # 1. GPU Transcription
    yield "Transcribing audio... please wait.", "Analysis will appear here...", audio_path
    scribe_output = scribe_audio(audio_path)
    full_transcript = scribe_output["text"]
    
    # 2. Chunking
    words = full_transcript.split()
    step = max(1, int(chunk_size - overlap))
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + int(chunk_size)]))
        if i + int(chunk_size) >= len(words):
            break

    # 3. GPU Analysis (Streaming back to UI)
    final_report = []
    for i, chunk_text in enumerate(chunks):
        # Update UI to show progress
        yield full_transcript, f"Analyzing part {i+1} of {len(chunks)}...", audio_path
        
        analysis = generate(chunk_text)
        final_report.append(f"### PART {i+1} ANALYSIS\n{analysis}")
        
        # Immediately show what we have so far
        current_report = "\n\n".join(final_report)
        yield full_transcript, current_report, audio_path


# --- UI Layout ---
with gr.Blocks(fill_height=True, fill_width=True) as demo:
    gr.Markdown("# 🇦🇷 Spanish Localizer")
    
    with gr.Row():
        with gr.Column(scale=1): 
            audio_input = gr.File(label="Upload MP3/MP4", file_types=[".mp3", ".mp4", ".wav"])
            
            with gr.Row():
                chunk_slider = gr.Slider(minimum=200, maximum=2000, value=1000, step=100, label="Chunk Size (Words)")
                overlap_slider = gr.Slider(minimum=0, maximum=400, value=150, step=50, label="Overlap (Words)")
            
            audio_btn = gr.Button("Transcribe & Analyze Audio", variant="primary")
            debug_audio = gr.Audio(label="Extracted Audio", type="filepath")

        with gr.Column(scale=1):
            transcript_box = gr.Textbox(label="Transcription", interactive=False, lines=8)
            output_box = gr.Textbox(label="Argentinian Analysis", interactive=False, lines=20, scale=1)

    audio_btn.click(
        fn=process_audio, 
        inputs=[audio_input, chunk_slider, overlap_slider], 
        outputs=[transcript_box, output_box, debug_audio]
    )

demo.launch()
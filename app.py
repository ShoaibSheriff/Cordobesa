import spaces
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os 
from huggingface_hub import login 
import librosa
from moviepy import VideoFileClip
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Pipeline 
from pyannote.core import Segment
import shutil


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

#scribe_model = whisper.load_model("large-v3-turbo")

scribe_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", chunk_length_s=30, torch_dtype=torch.float16, device="cuda")

def prepare_audio(file_path):
    if file_path.endswith((".mp4", ".mov")):
        print("Extracting audio on CPU...")
        video = VideoFileClip(file_path)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        # CPU handles the conversion to 16kHz WAV
        video.audio.write_audiofile(temp_audio, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
        
        # Move the temp file to your current directory so you can see it in the file explorer
        shutil.copy(temp_audio, "check_this_audio.wav")
        
        return temp_audio
    return file_path

@spaces.GPU(duration=120)
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


embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def semantic_topic_chunks(text, percentile_threshold=80):
    # 1. Split into sentences or speaker turns
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 2:
        return [text]

    # 2. Get embeddings for every line
    embeddings = embed_model.encode(lines)
    
    # 3. Calculate 'distances' (similarity drops) between consecutive lines
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)
    
    # 4. Identify the biggest "drops" in similarity (the topic shifts)
    # A drop in similarity means the conversation changed direction
    threshold = np.percentile(similarities, 100 - percentile_threshold)
    
    chunks = []
    current_chunk = [lines[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # Topic shift detected! Start a new chunk
            chunks.append("\n".join(current_chunk))
            current_chunk = [lines[i+1]]
        else:
            current_chunk.append(lines[i+1])
            
    chunks.append("\n".join(current_chunk))
    return chunks


def align_speakers(whisper_results, diarization_output):
    aligned_lines = []
    
    for chunk in whisper_results["chunks"]:
        # Get Whisper's timing for this piece of text
        start_t, end_t = chunk["timestamp"]
        text = chunk["text"].strip()
        
        # Create a Pyannote Segment for this chunk
        whisper_segment = Segment(start_t, end_t)
        
        # Track which speaker has the most 'airtime' in this chunk
        speaker_durations = {}
        
        for turn, _, speaker in diarization_output.itertracks(yield_label=True):
            # Calculate intersection between Whisper chunk and Speaker turn
            intersection = turn & whisper_segment
            if intersection:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + intersection.duration
        
        # Pick the winner, or fallback to 'Unknown' if no overlap found
        if speaker_durations:
            assigned_speaker = max(speaker_durations, key=speaker_durations.get)
        else:
            assigned_speaker = "Unknown"
            
        aligned_lines.append(f"[{assigned_speaker}]: {text}")
    
    return "\n".join(aligned_lines)


diarization_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token=os.getenv("HF_TOKEN") 
)
diarization_model.to(torch.device("cuda"))

@spaces.GPU(duration=120)
def scribe_audio(audio_path):

    diarization_op = diarization_model(audio_path, num_speakers=2)
    
    prompt_text = "Transcripción de una charla argentina con lunfardo y modismos de Buenos Aires."
    forced_prompt_ids = scribe_pipe.tokenizer.get_prompt_ids(prompt_text, return_tensors="pt").to("cuda")

    result = scribe_pipe(audio_path, return_timestamps=True, chunk_length_s=30, batch_size=8, generate_kwargs={"language":"spanish", "prompt_ids": forced_prompt_ids})

    # return align_speakers(result, diarization_op)
    return result

    

# --- UPDATE YOUR GENERATE FUNCTION ---
# We modify it to handle either raw text or audio

def process_audio(file_path):
    audio_path=prepare_audio(file_path)
    
    # Scribe logic with Argentinian context prompt
    # The initial_prompt helps Whisper expect 'sh' sounds and slang
    #result = scribe_model.transcribe(
    #    audio_path, 
    #    initial_prompt="Transcripción de una charla argentina con lunfardo y modismos de Buenos Aires."
    #)

    final_report= []

    full_text = scribe_audio(audio_path)

    # text_by_topic = semantic_topic_chunks(full_text)

    # for i, segment_text in enumerate(text_by_topic):
    #     print(f"Analyzing topic {i+1}")
    #     analysis = generate(segment_text)
    #     final_report.append(f"### TOPIC {i+1} ANALYSIS\n{analysis}")
    return full_text, "\n\n--\n\n".join(final_report)

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

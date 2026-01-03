import os
import torch
import gradio as gr
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from moviepy import VideoFileClip, AudioFileClip
import spaces


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

#scribe_model = whisper.load_model("large-v3-turbo")

scribe_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", chunk_length_s=30, torch_dtype=torch.float16, device="cuda")

def prepare_audio(file_path):
    if not file_path:
        return None
    
    # Handle Gradio file object
    actual_path = file_path.name if hasattr(file_path, 'name') else file_path
    
    print("Processing audio with FFmpeg normalization...")
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    if actual_path.lower().endswith((".mp4", ".mov", ".mkv")):
        video = VideoFileClip(actual_path)
        # Normalize and convert to 16kHz Mono (WhisperX Requirement)
        video.audio.write_audiofile(
            temp_audio, 
            fps=16000, 
            nbytes=2, 
            codec='pcm_s16le', 
            ffmpeg_params=["-ac", "1", "-af", "loudnorm=I=-20:TP=-1.5:LRA=11"],
            logger=None
        )
        video.close()
    else:
        # Even if it's already audio, we normalize it for better diarization
        from moviepy.editor import AudioFileClip
        audio = AudioFileClip(actual_path)
        audio.write_audiofile(
            temp_audio, 
            fps=16000, 
            nbytes=2, 
            codec='pcm_s16le', 
            ffmpeg_params=["-ac", "1", "-af", "loudnorm=I=-20:TP=-1.5:LRA=11"],
            logger=None
        )
        audio.close()
        
    return temp_audio


@spaces.GPU(duration=120)
def generate(text):
    # Added context to the prompt so the model knows it's a segment
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
    
    # Using bfloat16 and SDPA for speed on 8B model
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        do_sample=False,
        temperature=0.0 # Keep it precise for translation
    )
    
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

    result = scribe_pipe(audio_path, return_timestamps=False, chunk_length_s=30, stride_length_s=5, batch_size=8, generate_kwargs={"do_sample": True,                  # Add slight randomness to break loops
        "temperature": 0.2, "no_repeat_ngram_size": 6, "language":"spanish", "condition_on_prev_tokens": False, "prompt_ids": forced_prompt_ids, "no_speech_threshold": 0.6, # 0.6 is a good balance for noisy audio
        "logprob_threshold": -1.0  })

    # return align_speakers(result, diarization_op)
    return result

    

# --- UPDATE YOUR GENERATE FUNCTION ---
# We modify it to handle either raw text or audio

def process_audio(file_path):
    if not file_path:
        return None, "No file uploaded.", None
        
    audio_path = prepare_audio(file_path)
    
    # 1. Get the full transcription
    # Note: Ensure scribe_audio returns a string, not a dict
    full_transcript = scribe_audio(audio_path)["text"]
    
    # 2. Chunking Logic (Recursive with Overlap)
    # We target ~1500 tokens per chunk to leave room for the analysis output
    words = full_transcript.split()
    chunk_size = 1500
    overlap = 200
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    # 3. Process each chunk through Llama-3
    final_report = []
    for i, chunk_text in enumerate(chunks):
        print(f"Analyzing chunk {i+1}/{len(chunks)}...")
        analysis = generate(chunk_text)
        final_report.append(f"### PART {i+1} ANALYSIS\n{analysis}")

    # Join results with a visual separator
    full_analysis = "\n\n" + "="*30 + "\n\n".join(final_report)
    
    return full_transcript, full_analysis, audio_path

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

        # ADD THIS: The Audio Player for debugging
        debug_audio = gr.Audio(label="Extracted Audio (Check for glitches here)", type="filepath")

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
        outputs=[transcript_box, output_box, debug_audio] # Added debug_audio here
    )

# Final step: launch the demo that was defined in the 'with' block
demo.launch()

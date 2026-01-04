# &#x1f1e6; Rioplatense Decoder ➔ 🇬🇧 English Analysis
### *Transcription & Sociolinguistic Localization Pipeline*

The **Rioplatense Decoder** is a high-performance AI pipeline designed to transcribe, translate, and culturally decode the rich, complex dialects of Argentina (specifically "Porteño" Spanish). Standard translation tools often fail to capture the nuances of *lunfardo* (slang) and the specific emotional weight of Rioplatense idioms. This application solves that by combining high-fidelity speech recognition with a specialized LLM "expert" in Argentinian culture.

---

### 🚀 Technical Architecture

This system utilizes a **Sliding Window Generator** architecture to process long-form audio while maintaining high context precision and avoiding hardware timeouts on GPU-hosted environments.



1.  **Audio Normalization (🇦🇷 ➔ 🎧):** Raw audio or video is extracted via **FFmpeg/MoviePy** and passed through a `loudnorm` filter. This standardizes the signal to **-20 LUFS**, ensuring the transcription engine receives a clear, consistent mono signal.
2.  **Dialect-Aware Transcription (🎧 ➔ 📝):** Using **OpenAI’s Whisper-v3-Turbo**, we "prime" the model with a specific Argentinian phonetic prompt. This significantly improves accuracy for regionalisms like *shaismo* (the specific 'sh' sound for *y* and *ll*).
3.  **Sociolinguistic Chunking:** To maintain context, the transcript is sliced into overlapping segments. This ensures that an idiom or cultural reference spanning two sentences is never "cut in half" during analysis.
    
4.  **Linguistic Decoding (📝 ➔ 🧠):** Powered by **Llama-3.1-8B-Instruct**, each segment is analyzed for:
    * **Lunfardo:** Contextualizing street slang (e.g., *bondi*, *guita*, *pibe*).
    * **Cultural Metaphors:** Explaining the intent behind regional expressions.
5.  **English Localization (🧠 ➔ 🇬🇧):** A final translation layer focuses on "vibe-accuracy" and equivalence of intent rather than literal word-for-word substitution.

---

### ✨ Key Features

* **Streaming Results:** Uses a Python Generator pattern to `yield` analysis part-by-part, so the UI updates in real-time.
* **Heartbeat Architecture:** Optimized for ZeroGPU environments to prevent proxy token expiration during heavy inference tasks.
* **Interactive Control:** Dynamic sliders for **Chunk Size** and **Overlap** allow users to tune the granularity of the report.
* **Visual Debugging:** Integrated audio player to check the processed/normalized audio used by the AI.



---

### 🛠️ Installation & Usage

1.  **Clone & Install:**
    ```bash
    git clone [https://github.com/your-username/rioplatense-decoder.git](https://github.com/your-username/rioplatense-decoder.git)
    pip install -r requirements.txt
    ```
2.  **Environment Variables:**
    Set your `HF_TOKEN` in your environment to access the Llama-3.1 gated weights.
3.  **Run:**
    ```bash
    python app.py
    ```

---

### 📖 Usage Guide

1.  **Upload:** Drop an **🇦🇷 .mp3, .mp4, or .wav** file.
2.  **Tune:** Use the sliders to set your preferred analysis window.
3.  **Process:** Click **"Transcribe & Analyze"**.
4.  **Review:** Watch the transcription and English analysis stream live in the output columns.

---

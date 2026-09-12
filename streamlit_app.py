import os
import re
import uuid
import asyncio
import requests
import pymupdf
import edge_tts
import streamlit as st
from google import genai
from google.genai import types
from pydub import AudioSegment
from pydub.generators import Sine

# =========================================================
# KONFIGURASI DASAR
# =========================================================
st.set_page_config(
    page_title="Executive Academic Reader & Research Companion",
    page_icon="🎧",
    layout="wide",
)

BACKGROUND_OPTIONS = [
    "Tanpa Backsound (Hanya Vokal)",
    "Piano 1: Relaxing C-Major (Soft)",
    "Piano 2: Deep Focus A-Minor (Calm)",
    "Piano 3: Gentle F-Major (Warm)",
]

ALL_SECTIONS = [
    "Abstract", "Problem", "Pains", "Needs", "Importants", "Solution",
    "Contribution", "Limitation and Future Works", "Previous Works",
    "Hypothesis", "Research Question", "Overview Methods",
    "Proposed Method in Detail", "Experiments", "Discussion", "Findings",
]

OUTPUT_DIR = "audio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AppError(Exception):
    """Error yang ditampilkan langsung ke user lewat st.error()."""
    pass


# Setiap sesi browser mendapat ID unik supaya file audio antar-user tidak bentrok
# saat aplikasi diakses banyak orang secara bersamaan.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

SID = st.session_state.session_id


# =========================================================
# FUNGSI INTI (LOGIKA TIDAK DIUBAH DARI VERSI GRADIO)
# =========================================================
def clean_text(text: str) -> str:
    ref_pattern = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)
    match = ref_pattern.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def extract_text_from_input(uploaded_file, url_or_doi: str) -> str:
    """
    uploaded_file: objek dari st.file_uploader (punya .name dan .getvalue()),
    berbeda dari Gradio yang punya .name sebagai path di disk.
    """
    text = ""

    if url_or_doi and url_or_doi.strip():
        target = url_or_doi.strip()
        if target.startswith("10."):
            target = f"https://doi.org/{target}"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            res = requests.get(target, headers=headers, timeout=25)
            if res.status_code == 200:
                if "application/pdf" in res.headers.get("Content-Type", "") or target.endswith(".pdf"):
                    doc = pymupdf.open(stream=res.content, filetype="pdf")
                    for page in doc:
                        text += page.get_text("text") + "\n"
                else:
                    text = res.text
            else:
                raise AppError(f"Gagal mengakses URL/DOI (Status: {res.status_code}).")
        except AppError:
            raise
        except Exception as e:
            raise AppError(f"Gagal mengambil data dari URL/DOI: {str(e)}")

    elif uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        data = uploaded_file.getvalue()

        if ext == ".pdf":
            doc = pymupdf.open(stream=data, filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
        elif ext in [".md", ".txt"]:
            text = data.decode("utf-8", errors="ignore")
        else:
            raise AppError("Format file tidak didukung! Unggah file .pdf, .md, atau .txt.")
    else:
        raise AppError("Silakan unggah berkas paper ATAU masukkan URL/DOI paper!")

    text = clean_text(text)
    if not text:
        raise AppError("Tidak dapat mengekstrak teks dari input yang diberikan.")

    return text


def create_piano_ambiance(track_type: str, duration_ms: int):
    if track_type == "Tanpa Backsound (Hanya Vokal)" or duration_ms <= 0:
        return None

    notes_map = {
        "Piano 1: Relaxing C-Major (Soft)": [261.63, 329.63, 392.00, 523.25],
        "Piano 2: Deep Focus A-Minor (Calm)": [220.00, 261.63, 329.63, 440.00],
        "Piano 3: Gentle F-Major (Warm)": [174.61, 220.00, 261.63, 349.23],
    }

    frequencies = notes_map.get(track_type, notes_map["Piano 1: Relaxing C-Major (Soft)"])
    combined_track = AudioSegment.silent(duration=duration_ms)

    for freq in frequencies:
        sine_wave = Sine(freq).to_audio_segment(duration=duration_ms, volume=-28)
        sine_wave = sine_wave.fade_in(1500).fade_out(2000)
        combined_track = combined_track.overlay(sine_wave)

    return combined_track


def mix_vocal_and_music(vocal_path: str, selected_music: str) -> str:
    if not vocal_path or not os.path.exists(vocal_path):
        raise AppError("Audio vokal belum tersedia. Silakan proses paper terlebih dahulu!")

    output_path = vocal_path.replace(".mp3", "_mixed.mp3")
    vocal_track = AudioSegment.from_file(vocal_path)
    duration_ms = len(vocal_track)

    if selected_music == "Tanpa Backsound (Hanya Vokal)":
        vocal_track.export(output_path, format="mp3")
        return output_path

    try:
        bg_music = create_piano_ambiance(selected_music, duration_ms)
        if bg_music:
            bg_music = bg_music - 20
            final_mix = vocal_track.overlay(bg_music)
            final_mix.export(output_path, format="mp3")
        else:
            vocal_track.export(output_path, format="mp3")
        return output_path
    except Exception:
        vocal_track.export(output_path, format="mp3")
        return output_path


async def generate_neural_audio(text: str, voice_code: str, output_file: str):
    communicate = edge_tts.Communicate(text, voice_code)
    await communicate.save(output_file)


def get_api_key(api_key_input: str) -> str:
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()

    # Urutan pencarian fallback: st.secrets (Streamlit Cloud / secrets.toml lokal) -> environment variable
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret_key = None

    effective = secret_key or os.getenv("GEMINI_API_KEY")
    if not effective:
        raise AppError("Masukkan Gemini API Key!")
    return effective


def run_tts(script: str, language: str, tab_name: str) -> str:
    voice_code = "id-ID-ArdiNeural" if language == "Bahasa Indonesia" else "en-US-ChristopherNeural"
    vocal_path = os.path.join(OUTPUT_DIR, f"{tab_name}_vocal_{SID}.mp3")
    asyncio.run(generate_neural_audio(script, voice_code, vocal_path))
    return vocal_path


# =========================================================
# PROSES TIAP TAB (LOGIKA SAMA DENGAN app.py ASLI)
# =========================================================
def process_tab1(uploaded_file, url_input, api_key, language):
    effective_api_key = get_api_key(api_key)
    raw_text = extract_text_from_input(uploaded_file, url_input)
    client = genai.Client(api_key=effective_api_key)

    lang_instruction = "Gunakan BAHASA INDONESIA penuh." if language == "Bahasa Indonesia" else "Use ENGLISH language as in original paper."

    system_prompt = f"""
Anda adalah narator akademis profesional.
Tugas Anda adalah merangkum isi paper ilmiah berikut menjadi teks pembacaan audio yang sangat padat, ringkas, lancar, dan berintonasi alami seperti manusia membaca artikel secara langsung.
{lang_instruction}

KANDUNGAN MATERI:
1. Problem utama & research gap.
2. Hipotesis penelitian (formulasikan secara tersirat jika tidak tertulis eksplisit).
3. Kontribusi konkret penelitian.
4. Garis besar metode yang diusulkan serta perbandingannya dengan SOTA.

ATURAN SINTESIS TEKS:
1. TANPA SALAM PEMBUKA.
2. TANPA FORMAT MARKDOWN (*, #, -, _). Tuliskan dalam paragraf naratif mengalir.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[system_prompt, f"Teks paper:\n\n{raw_text}"],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    script = response.text
    vocal_path = run_tts(script, language, "tab1")
    return script, vocal_path


def process_tab2(uploaded_file, url_input, api_key, selected_sections, language):
    effective_api_key = get_api_key(api_key)
    if not selected_sections:
        raise AppError("Pilih minimal 1 bagian artikel!")

    raw_text = extract_text_from_input(uploaded_file, url_input)
    client = genai.Client(api_key=effective_api_key)

    sections_str = ", ".join(selected_sections)
    lang_instruction = "Gunakan BAHASA INDONESIA penuh." if language == "Bahasa Indonesia" else "Use ENGLISH language as in original paper."

    system_prompt = f"""
Anda adalah narator akademis tingkat lanjut.
Tugas Anda adalah menyusun narasi audio dari paper ilmiah HANYA untuk bagian-bagian terpilih berikut: [{sections_str}].
{lang_instruction}

INSTRUKSI THINKING AI:
- Jika aspek yang dicentang tidak tertulis secara eksplisit dalam teks paper, gunakan penalaran mendalam (deep deduction) berbasis konteks keseluruhan isi paper untuk merumuskan poin tersebut secara akurat dan logis.

ATURAN SINTESIS TEKS:
1. TANPA SALAM PEMBUKA.
2. DILARANG GUNAKAN MARKDOWN (*, #, -, _). Tuliskan dalam narasi paragraf mengalir.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[system_prompt, f"Teks paper:\n\n{raw_text}"],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    script = response.text
    vocal_path = run_tts(script, language, "tab2")
    return script, vocal_path


def process_tab3(uploaded_file, url_input, api_key, language):
    effective_api_key = get_api_key(api_key)
    raw_text = extract_text_from_input(uploaded_file, url_input)
    client = genai.Client(api_key=effective_api_key)

    lang_instruction = "Gunakan BAHASA INDONESIA penuh." if language == "Bahasa Indonesia" else "Use ENGLISH language as in original paper."

    system_prompt = f"""
Anda adalah narator dan pembaca naskah profesional.
Tugas Anda adalah membaca dan menyampaikan KESELURUHAN isi paper ilmiah ini secara utuh, runtut, mendetail, dan lengkap.
{lang_instruction}

ATURAN SINTESIS TEKS:
1. TANPA SALAM PEMBUKA.
2. DILARANG KERAS menggunakan format Markdown (*, #, -, _). Tuliskan dalam bentuk paragraf naratif utuh.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[system_prompt, f"Teks paper lengkap:\n\n{raw_text}"],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    script = response.text
    vocal_path = run_tts(script, language, "tab3")
    return script, vocal_path


def chat_with_paper_ai(user_message, chat_history, uploaded_file, url_input, api_key, discussion_mode):
    effective_api_key = get_api_key(api_key)

    try:
        raw_text = extract_text_from_input(uploaded_file, url_input)
    except AppError as e:
        raise AppError(f"Sediakan dokumen/URL paper terlebih dahulu! Detail: {str(e)}")

    client = genai.Client(api_key=effective_api_key)

    if discussion_mode == "Pendalaman Materi Paper":
        system_instruction = "Anda adalah pakar akademis dan tutor riset. Jawab pertanyaan pengguna tentang isi paper ini dengan runtut, akurat, dan intuitif."
    else:
        system_instruction = "Anda adalah Peneliti Utama dan reviewer jurnal. Gunakan penalaran kritis untuk menemukan keterbatasan riset (research gaps) dan memberikan ide pengembangan riset selanjutnya."

    conversation_context = ""
    for user_turn, ai_turn in chat_history:
        conversation_context += f"User: {user_turn}\nAI: {ai_turn}\n"

    full_prompt = f"""
{system_instruction}

TEKS REFERENSI PAPER:
---
{raw_text[:30000]}
---

RIWAYAT DISKUSI:
{conversation_context}

PERTANYAAN USER:
{user_message}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[full_prompt],
        config=types.GenerateContentConfig(temperature=0.3),
    )

    return response.text


# =========================================================
# ANTAR MUKA STREAMLIT
# =========================================================
st.title("🎧 Executive Academic Reader & Research Companion")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Seluruh Artikel", "Pilih Bagian", "Diskusi AI"])

# ---------------- TAB 1: RINGKASAN ----------------
with tab1:
    st.markdown("### 📄 Ringkasan Eksekutif Paper")
    col1, col2 = st.columns(2)

    with col1:
        api_key_t1 = st.text_input("Gemini API Key (Opsional)", type="password", placeholder="AIzaSy...", key="key1")
        pdf_t1 = st.file_uploader("1A. Unggah Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key="file1")
        url_t1 = st.text_input("1B. Atau Tempel URL / DOI Paper", placeholder="https://arxiv.org/pdf/... atau 10.1016/...", key="url1")
        lang_t1 = st.radio("Pilihan Bahasa", ["Bahasa Indonesia", "English / Original Language"], key="lang1")
        if st.button("🚀 1. Proses & Buat Narasi Ringkasan", type="primary", key="btn1"):
            try:
                with st.spinner("Memproses paper dan membuat narasi..."):
                    script, vocal_path = process_tab1(pdf_t1, url_t1, api_key_t1, lang_t1)
                st.session_state.script_t1 = script
                st.session_state.vocal_t1 = vocal_path
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    with col2:
        st.text_area("📄 Teks Narasi Ringkasan", value=st.session_state.get("script_t1", ""), height=250, key="script_area1")
        st.markdown("---")
        bg_t1 = st.selectbox("Pilih Musik Backsound", BACKGROUND_OPTIONS, key="bg1")
        if st.button("🎵 2. Gabungkan / Update Audio Backsound", key="mixbtn1"):
            try:
                mixed_path = mix_vocal_and_music(st.session_state.get("vocal_t1"), bg_t1)
                st.session_state.mixed_t1 = mixed_path
            except AppError as e:
                st.error(str(e))
        play_path = st.session_state.get("mixed_t1") or st.session_state.get("vocal_t1")
        if play_path and os.path.exists(play_path):
            st.audio(play_path)

# ---------------- TAB 2: SELURUH ARTIKEL (BAGIAN TERPILIH) ----------------
with tab2:
    st.markdown("### 🔍 Narasi Bagian-Bagian Spesifik Artikel")
    col1, col2 = st.columns(2)

    with col1:
        api_key_t2 = st.text_input("Gemini API Key (Opsional)", type="password", placeholder="AIzaSy...", key="key2")
        pdf_t2 = st.file_uploader("1A. Unggah Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key="file2")
        url_t2 = st.text_input("1B. Atau Tempel URL / DOI Paper", placeholder="https://arxiv.org/pdf/...", key="url2")
        lang_t2 = st.radio("Pilihan Bahasa", ["Bahasa Indonesia", "English / Original Language"], key="lang2")
        sections_cb = st.multiselect("Pilih Bagian Artikel", ALL_SECTIONS, default=["Abstract"], key="sections2")
        if st.button("🚀 1. Buat Narasi Bagian Terpilih", type="primary", key="btn2"):
            try:
                with st.spinner("Memproses paper dan membuat narasi..."):
                    script, vocal_path = process_tab2(pdf_t2, url_t2, api_key_t2, sections_cb, lang_t2)
                st.session_state.script_t2 = script
                st.session_state.vocal_t2 = vocal_path
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    with col2:
        st.text_area("📄 Teks Narasi Bagian Terpilih", value=st.session_state.get("script_t2", ""), height=250, key="script_area2")
        st.markdown("---")
        bg_t2 = st.selectbox("Pilih Musik Backsound", BACKGROUND_OPTIONS, key="bg2")
        if st.button("🎵 2. Gabungkan / Update Audio Backsound", key="mixbtn2"):
            try:
                mixed_path = mix_vocal_and_music(st.session_state.get("vocal_t2"), bg_t2)
                st.session_state.mixed_t2 = mixed_path
            except AppError as e:
                st.error(str(e))
        play_path = st.session_state.get("mixed_t2") or st.session_state.get("vocal_t2")
        if play_path and os.path.exists(play_path):
            st.audio(play_path)

# ---------------- TAB 3: PEMBACAAN UTUH ----------------
with tab3:
    st.markdown("### 📖 Pembacaan Keseluruhan Isi Paper (Full Reader)")
    col1, col2 = st.columns(2)

    with col1:
        api_key_t3 = st.text_input("Gemini API Key (Opsional)", type="password", placeholder="AIzaSy...", key="key3")
        pdf_t3 = st.file_uploader("1A. Unggah Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key="file3")
        url_t3 = st.text_input("1B. Atau Tempel URL / DOI Paper", placeholder="https://...", key="url3")
        lang_t3 = st.radio("Pilihan Bahasa", ["Bahasa Indonesia", "English / Original Language"], key="lang3")
        if st.button("🚀 1. Proses Pembacaan Utuh Paper", type="primary", key="btn3"):
            try:
                with st.spinner("Memproses paper dan membuat narasi..."):
                    script, vocal_path = process_tab3(pdf_t3, url_t3, api_key_t3, lang_t3)
                st.session_state.script_t3 = script
                st.session_state.vocal_t3 = vocal_path
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    with col2:
        st.text_area("📄 Teks Narasi Utuh Keseluruhan Paper", value=st.session_state.get("script_t3", ""), height=250, key="script_area3")
        st.markdown("---")
        bg_t3 = st.selectbox("Pilih Musik Backsound", BACKGROUND_OPTIONS, key="bg3")
        if st.button("🎵 2. Gabungkan / Update Audio Backsound", key="mixbtn3"):
            try:
                mixed_path = mix_vocal_and_music(st.session_state.get("vocal_t3"), bg_t3)
                st.session_state.mixed_t3 = mixed_path
            except AppError as e:
                st.error(str(e))
        play_path = st.session_state.get("mixed_t3") or st.session_state.get("vocal_t3")
        if play_path and os.path.exists(play_path):
            st.audio(play_path)

# ---------------- TAB 4: DISKUSI CHATBOT ----------------
with tab4:
    st.markdown("### 💡 Diskusi & Tanya Jawab Interaktif tentang Paper")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (user, ai) tuples

    col1, col2 = st.columns([1, 2])

    with col1:
        api_key_t4 = st.text_input("Gemini API Key (Opsional)", type="password", placeholder="AIzaSy...", key="key4")
        pdf_t4 = st.file_uploader("1A. Unggah Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key="file4")
        url_t4 = st.text_input("1B. Atau Tempel URL / DOI Paper", placeholder="https://...", key="url4")
        chat_mode = st.radio(
            "Fokus Diskusi AI",
            ["Pendalaman Materi Paper", "Mencari Research Gap & Pengisian Celah Riset"],
            key="chatmode4",
        )
        if st.button("🗑️ Bersihkan Obrolan", key="clearbtn4"):
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        chat_box = st.container(height=400)
        with chat_box:
            for user_turn, ai_turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(user_turn)
                with st.chat_message("assistant"):
                    st.write(ai_turn)

        user_message = st.chat_input("Ketik pertanyaan Anda... (mis. Bisakah jelaskan persamaan (3)?)", key="chatinput4")
        if user_message:
            try:
                with st.spinner("Gemini sedang berpikir..."):
                    ai_reply = chat_with_paper_ai(
                        user_message, st.session_state.chat_history, pdf_t4, url_t4, api_key_t4, chat_mode
                    )
                st.session_state.chat_history.append((user_message, ai_reply))
                st.rerun()
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

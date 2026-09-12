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

st.set_page_config(page_title="Paper Audio Reader", layout="wide")

NO_BACKGROUND = "Tanpa Backsound"
BACKGROUND_OPTIONS = [
    NO_BACKGROUND,
    "Piano 1: Relaxing C-Major",
    "Piano 2: Deep Focus A-Minor",
    "Piano 3: Gentle F-Major",
]

PIANO_NOTES = {
    "Piano 1: Relaxing C-Major": [261.63, 329.63, 392.00, 523.25],
    "Piano 2: Deep Focus A-Minor": [220.00, 261.63, 329.63, 440.00],
    "Piano 3: Gentle F-Major": [174.61, 220.00, 261.63, 349.23],
}

ALL_SECTIONS = [
    "Abstract", "Problem", "Pains", "Needs", "Importants", "Solution",
    "Contribution", "Limitation and Future Works", "Previous Works",
    "Hypothesis", "Research Question", "Overview Methods",
    "Proposed Method in Detail", "Experiments", "Discussion", "Findings",
]

LANGUAGES = ["Bahasa Indonesia", "English / Original Language"]
VOICE_MAP = {
    "Bahasa Indonesia": "id-ID-ArdiNeural",
    "English / Original Language": "en-US-ChristopherNeural",
}

CHAT_MODES = {
    "Pendalaman Materi": "Anda pakar akademis dan tutor riset. Jawab pertanyaan tentang isi paper ini dengan runtut dan akurat.",
    "Cari Research Gap": "Anda peneliti utama dan reviewer jurnal. Gunakan penalaran kritis untuk menemukan keterbatasan riset dan ide pengembangan selanjutnya.",
}

OUTPUT_DIR = "audio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AppError(Exception):
    pass


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
SID = st.session_state.session_id


def clean_text(text):
    match = re.search(r"\n\s*(references|bibliography)\s*\n", text, re.IGNORECASE)
    return (text[: match.start()] if match else text).strip()


def pdf_to_text(data):
    doc = pymupdf.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)


def fetch_url_text(target):
    if target.startswith("10."):
        target = f"https://doi.org/{target}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(target, headers=headers, timeout=25)
    except Exception as e:
        raise AppError(f"Gagal mengambil data dari URL/DOI: {e}")
    if res.status_code != 200:
        raise AppError(f"Gagal mengakses URL/DOI (status {res.status_code}).")
    if "application/pdf" in res.headers.get("Content-Type", "") or target.endswith(".pdf"):
        return pdf_to_text(res.content)
    return res.text


def extract_text(uploaded_file, url_or_doi):
    if url_or_doi and url_or_doi.strip():
        raw = fetch_url_text(url_or_doi.strip())
    elif uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".pdf":
            raw = pdf_to_text(uploaded_file.getvalue())
        elif ext in (".md", ".txt"):
            raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        else:
            raise AppError("Format tidak didukung. Gunakan .pdf, .md, atau .txt.")
    else:
        raise AppError("Unggah berkas paper atau masukkan URL/DOI.")

    text = clean_text(raw)
    if not text:
        raise AppError("Tidak dapat mengekstrak teks dari input yang diberikan.")
    return text


def build_ambiance(track, duration_ms):
    if track == NO_BACKGROUND or duration_ms <= 0:
        return None
    frequencies = PIANO_NOTES[track]
    layer = AudioSegment.silent(duration=duration_ms)
    for freq in frequencies:
        tone = Sine(freq).to_audio_segment(duration=duration_ms, volume=-28)
        tone = tone.fade_in(1500).fade_out(2000)
        layer = layer.overlay(tone)
    return layer


def mix_audio(vocal_path, background):
    if not vocal_path or not os.path.exists(vocal_path):
        raise AppError("Audio vokal belum tersedia. Proses paper terlebih dahulu.")

    # 呼ビョンマダ アウトプット トポッソ, ミクス ボタン ヌルル タダ サセキ ファイル
    output_path = vocal_path.replace(".mp3", "_mixed.mp3")
    vocal = AudioSegment.from_file(vocal_path)
    ambiance = build_ambiance(background, len(vocal))
    mixed = vocal.overlay(ambiance - 20) if ambiance else vocal
    mixed.export(output_path, format="mp3")
    return output_path


async def _synthesize(text, voice, path):
    await edge_tts.Communicate(text, voice).save(path)


def narrate(text, language, tag):
    path = os.path.join(OUTPUT_DIR, f"{tag}_{SID}.mp3")
    asyncio.run(_synthesize(text, VOICE_MAP[language], path))
    return path


def resolve_api_key(user_input):
    if user_input and user_input.strip():
        return user_input.strip()

    # モンジョ セクレッツ プター, オプスミョン env カジョワ タ
    try:
        secret = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        secret = None

    key = secret or os.getenv("GEMINI_API_KEY")
    if not key:
        raise AppError("Masukkan Gemini API Key.")
    return key


def ask_gemini(api_key, prompt, source_text="", temperature=0.2):
    client = genai.Client(api_key=api_key)
    contents = [prompt, source_text] if source_text else [prompt]
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def lang_instruction(language):
    return "Gunakan BAHASA INDONESIA penuh." if language == "Bahasa Indonesia" else "Use ENGLISH language as in original paper."


def summary_prompt(language):
    return f"""
Anda adalah narator akademis profesional.
Tugas Anda adalah merangkum isi paper ilmiah berikut menjadi teks pembacaan audio yang sangat padat, ringkas, lancar, dan berintonasi alami seperti manusia membaca artikel secara langsung.
{lang_instruction(language)}

KANDUNGAN MATERI:
1. Problem utama & research gap.
2. Hipotesis penelitian (formulasikan secara tersirat jika tidak tertulis eksplisit).
3. Kontribusi konkret penelitian.
4. Garis besar metode yang diusulkan serta perbandingannya dengan SOTA.

ATURAN SINTESIS TEKS:
1. WAJIB DISEBUT DI AWAL: Mulai narasi pembacaan audio secara eksplisit dengan sapaan hormat: "Yang mulia, King paduka maha raja yang terhormat...".
2. TANPA FORMAT MARKDOWN (*, #, -, _). Tuliskan dalam paragraf naratif mengalir.
"""


def sections_prompt(sections, language):
    sections_str = ", ".join(sections)
    return f"""
Anda adalah narator akademis tingkat lanjut.
Tugas Anda adalah menyusun narasi audio dari paper ilmiah HANYA untuk bagian-bagian terpilih berikut: [{sections_str}].
{lang_instruction(language)}

INSTRUKSI THINKING AI:
- Jika aspek yang dicentang tidak tertulis secara eksplisit dalam teks paper, gunakan penalaran mendalam (deep deduction) berbasis konteks keseluruhan isi paper untuk merumuskan poin tersebut secara akurat dan logis.

ATURAN SINTESIS TEKS:
1. WAJIB DISEBUT DI AWAL: Mulai narasi pembacaan audio secara eksplisit dengan sapaan hormat: "Yang mulia, King paduka maha raja yang terhormat...".
2. DILARANG GUNAKAN MARKDOWN (*, #, -, _). Tuliskan dalam narasi paragraf mengalir.
"""


def full_prompt(language):
    return f"""
Anda adalah narator dan pembaca naskah profesional.
Tugas Anda adalah membaca dan menyampaikan KESELURUHAN isi paper ilmiah ini secara utuh, runtut, mendetail, dan lengkap.
{lang_instruction(language)}

ATURAN SINTESIS TEKS:
1. WAJIB DISEBUT DI AWAL: Mulai narasi pembacaan audio secara eksplisit dengan sapaan hormat: "Yang mulia, King paduka maha raja yang terhormat...".
2. DILARANG KERAS menggunakan format Markdown (*, #, -, _). Tuliskan dalam bentuk paragraf naratif utuh.
"""


def process_summary(uploaded_file, url, api_key_input, language):
    api_key = resolve_api_key(api_key_input)
    text = extract_text(uploaded_file, url)
    script = ask_gemini(api_key, summary_prompt(language), f"Teks paper:\n\n{text}")
    return script, narrate(script, language, "summary")


def process_sections(uploaded_file, url, api_key_input, sections, language):
    if not sections:
        raise AppError("Pilih minimal satu bagian artikel.")
    api_key = resolve_api_key(api_key_input)
    text = extract_text(uploaded_file, url)
    script = ask_gemini(api_key, sections_prompt(sections, language), f"Teks paper:\n\n{text}")
    return script, narrate(script, language, "sections")


def process_full(uploaded_file, url, api_key_input, language):
    api_key = resolve_api_key(api_key_input)
    text = extract_text(uploaded_file, url)
    script = ask_gemini(api_key, full_prompt(language), f"Teks paper lengkap:\n\n{text}")
    return script, narrate(script, language, "full")


def chat_reply(message, history, uploaded_file, url, api_key_input, mode):
    api_key = resolve_api_key(api_key_input)
    try:
        text = extract_text(uploaded_file, url)
    except AppError as e:
        raise AppError(f"Sediakan dokumen/URL paper terlebih dahulu. {e}")

    history_text = "\n".join(f"User: {u}\nAI: {a}" for u, a in history)
    prompt = f"""
{CHAT_MODES[mode]}

TEKS REFERENSI PAPER:
---
{text[:30000]}
---

RIWAYAT DISKUSI:
{history_text}

PERTANYAAN USER:
{message}
"""
    return ask_gemini(api_key, prompt, temperature=0.3)


def render_reader_tab(key, process_fn, with_sections=False):
    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input("Gemini API Key", type="password", key=f"key_{key}")
        uploaded = st.file_uploader("Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key=f"file_{key}")
        url = st.text_input("URL / DOI", key=f"url_{key}")
        language = st.radio("Bahasa", LANGUAGES, key=f"lang_{key}")
        sections = st.multiselect("Bagian artikel", ALL_SECTIONS, default=["Abstract"], key=f"sections_{key}") if with_sections else None

        if st.button("Proses", type="primary", key=f"process_{key}"):
            try:
                with st.spinner("Memproses..."):
                    if with_sections:
                        script, vocal_path = process_fn(uploaded, url, api_key, sections, language)
                    else:
                        script, vocal_path = process_fn(uploaded, url, api_key, language)
                st.session_state[f"script_{key}"] = script
                st.session_state[f"vocal_{key}"] = vocal_path
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    with col2:
        st.text_area("Naskah", value=st.session_state.get(f"script_{key}", ""), height=250, key=f"script_area_{key}")
        background = st.selectbox("Musik latar", BACKGROUND_OPTIONS, key=f"bg_{key}")
        if st.button("Gabungkan audio", key=f"mix_{key}"):
            try:
                st.session_state[f"mixed_{key}"] = mix_audio(st.session_state.get(f"vocal_{key}"), background)
            except AppError as e:
                st.error(str(e))

        play_path = st.session_state.get(f"mixed_{key}") or st.session_state.get(f"vocal_{key}")
        if play_path and os.path.exists(play_path):
            st.audio(play_path)


def render_chat_tab():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 2])

    with col1:
        api_key = st.text_input("Gemini API Key", type="password", key="key_chat")
        uploaded = st.file_uploader("Berkas (.pdf, .md, .txt)", type=["pdf", "md", "txt"], key="file_chat")
        url = st.text_input("URL / DOI", key="url_chat")
        mode = st.radio("Fokus diskusi", list(CHAT_MODES.keys()), key="mode_chat")
        if st.button("Bersihkan", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        box = st.container(height=400)
        with box:
            for user_msg, ai_msg in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    st.write(ai_msg)

        message = st.chat_input("Tulis pertanyaan Anda", key="chat_input")
        if message:
            try:
                with st.spinner("Memproses..."):
                    reply = chat_reply(message, st.session_state.chat_history, uploaded, url, api_key, mode)
                st.session_state.chat_history.append((message, reply))
                st.rerun()
            except AppError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


st.title("Paper Audio Reader")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Pilih Bagian", "Seluruh Artikel", "Diskusi"])

with tab1:
    render_reader_tab("summary", process_summary)

with tab2:
    render_reader_tab("sections", process_sections, with_sections=True)

with tab3:
    render_reader_tab("full", process_full)

with tab4:
    render_chat_tab()

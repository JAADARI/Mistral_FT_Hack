import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch
import pytesseract
from PIL import Image
from io import BytesIO
import os
# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)
retrieved_jobs = client.jobs.retrieve(os.getenv("MISTRAL_JOB_ID"))
model = retrieved_jobs.fine_tuned_model

# Initialize Wav2Vec2 model and processor
tokenizer = Wav2Vec2CTCTokenizer(os.getenv("WAV2VEC2_TOKENIZER_PATH"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
processor = Wav2Vec2Processor.from_pretrained('boumehdi/wav2vec2-large-xlsr-moroccan-darija', tokenizer=tokenizer)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained('boumehdi/wav2vec2-large-xlsr-moroccan-darija')

# Function to transcribe audio using Wav2Vec2 model
def transcribe_audio_wav2vec2(file_path):
    input_audio, sr = librosa.load(file_path, sr=16000)
    input_values = processor(input_audio, return_tensors="pt", padding=True).input_values
    logits = wav2vec2_model(input_values).logits
    tokens = torch.argmax(logits, axis=-1)
    transcription = tokenizer.batch_decode(tokens)
    return transcription[0]

def summarize_text(text):
    prompt = (
        "مهمتك هي تلخيص نص مكتوب باللهجة المغاربية. "
        "لا يجوز لك استخدام المعلومات إلا من النص المقدم لك، "
        "ولا يجوز لك استخدام أي معلومات خارجية. "
        "قم بتلخيص النص في 30 كلمة على الأكثر باللغة المغاربية."
    )
    messages = [
        ChatMessage(role="assistant", content=prompt),
        ChatMessage(role="user", content=text),
    ]
    chat_response = client.chat(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content

def extract_text_from_image(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    text = pytesseract.image_to_string(img, lang='ara')
    return remove_special_characters(text)

def remove_special_characters(text: str) -> str:
    return text.replace('\n', "").replace("\x0c", "")

# Set up the Streamlit app
st.set_page_config(page_title="Voice and Image Input Summarizer", page_icon=":microphone:", layout="wide")
st.title("DarijaSummaScript : Enhancing summarization for OCR and ASR transcripts")

company_logo = "https://www.elyadata.com/assets/elyadata-logo.svg"
company_logo2 = "https://plugins.matomo.org/MistralAI/images/5.6.4/A_Mistral_AI_logo.png?w=1024"

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px 0;">
        <img src="{company_logo}" alt="Company Logo" style="width: 200px; margin-right: 20px;">
        <img src="{company_logo2}" alt="Company Logo2" style="width: 200px; margin-left: 20px;">
    </div>
    """,
    unsafe_allow_html=True
)


# Custom CSS for background and layout
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img.freepik.com/free-vector/low-poly-abstract-design_1048-11910.jpg?w=1380&t=st=1719587612~exp=1719588212~hmac=972f1ae9229eb869e4a87a83f5baaccf11efeadf39cbe8b7a2037e2b51ec9bb5");
    background-size: cover;
    background-position: center;  
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] {
    background: #f0f0f0;
}
.faded-box {
    background: linear-gradient(135deg, rgba(255, 175, 123, 0.8) 0%, rgba(215, 109, 119, 0.6) 74%, rgba(58, 28, 113, 0.4) 100%);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    color: navy;
    font-weight: bold;
}
.title {
    font-size: 2em;
    color: #333;
}
.sub-title {
    font-size: 1.2em;
    color: #666;
}
.section {
    margin-top: 30px;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# Project description
st.markdown(
    """
    <div class="faded-box">
        <p class="title"><strong>Project Description:</strong></p>
        <p>This application performs summarization of Aldarija text transcribed from audio using an Automatic Speech Recognition (ASR) model. 
        The summarization is carried out using a fine-tuned Mistral model on Aldarija data, which is particularly challenging due to the low-resource nature of the language.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload and record sections
st.markdown('<div class="sub-title">Upload your audio or image file:</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    uploaded_audio = st.file_uploader("Upload a WAV file", type="wav")
    uploaded_image = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png"])

with col2:
    st.write("Or record your audio (placeholder)")
    st.warning("Functionality not implemented yet.")

# Process uploaded audio file
if uploaded_audio is not None:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_audio.getbuffer())

    st.audio("uploaded_audio.wav")

    # Transcribe audio with progress bar
    with st.spinner('Transcribing...'):
        transcribed_text = transcribe_audio_wav2vec2("uploaded_audio.wav")
    st.success('Transcription complete!')

    # Display transcribed text
    st.markdown("### Transcribed Text:")
    st.markdown(f'<div class="faded-box">{transcribed_text}</div>', unsafe_allow_html=True)

    # Summarize transcribed text with progress bar
    with st.spinner('Summarizing...'):
        summary = summarize_text(transcribed_text)
    st.success('Summary generated!')

    # Display summary
    st.markdown("### Summary with Fine-tuned Mistral:")
    st.markdown(f'<div class="faded-box">{summary}</div>', unsafe_allow_html=True)

# Process uploaded image file
if uploaded_image is not None:
    img_bytes = uploaded_image.read()
    img = Image.open(BytesIO(img_bytes))
    st.image(img, caption='Uploaded Image')

    # Extract text from image with progress bar
    with st.spinner('Extracting text from image...'):
        extracted_text = extract_text_from_image(img_bytes)
    st.success('Text extraction complete!')

    # Display extracted text
    st.markdown("### Extracted Text:")
    st.markdown(f'<div class="faded-box">{extracted_text}</div>', unsafe_allow_html=True)

    # Summarize extracted text with progress bar
    with st.spinner('Summarizing...'):
        summary = summarize_text(extracted_text)
    st.success('Summary generated!')

    # Display summary
    st.markdown("### Summary with Fine-tuned Mistral:")
    st.markdown(f'<div class="faded-box">{summary}</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #555;
    }
    </style>
    <div class="footer">
    Built with ❤️ by ELYADATA Mistral Team
    </div>
    """,
    unsafe_allow_html=True
)

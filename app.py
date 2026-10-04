# pyrefly: ignore [missing-import]
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import textblob
from textblob import TextBlob
from nrclex import NRCLex
from transformers import pipeline
import language_tool_python

st.set_page_config(page_title="Text & Speech Analysis Hub", layout="wide")

# Download NLTK data for NRCLex if not already present
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    
# Initialize models/tools with caching to prevent reloading
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_spam_detector():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

@st.cache_resource
def load_grammar_tool():
    return language_tool_python.LanguageTool('en-US')

st.sidebar.title("Navigation")
features = [
    "Sentiment Analysis",
    "Speech-to-Text",
    "Text-to-Speech",
    "Emotion Detection",
    "Spam Detection",
    "Text Summarization",
    "Grammar Checking"
]
choice = st.sidebar.radio("Go to", features)

st.title(choice)

if choice == "Sentiment Analysis":
    st.write("Analyze the sentiment (polarity and subjectivity) of your text.")
    text_input = st.text_area("Enter text here:", height=150)
    if st.button("Analyze Sentiment"):
        if text_input:
            blob = TextBlob(text_input)
            st.write(f"**Polarity:** {blob.sentiment.polarity} (Range: -1.0 to 1.0)")
            st.write(f"**Subjectivity:** {blob.sentiment.subjectivity} (Range: 0.0 to 1.0)")
            if blob.sentiment.polarity > 0:
                st.success("Positive Sentiment")
            elif blob.sentiment.polarity < 0:
                st.error("Negative Sentiment")
            else:
                st.info("Neutral Sentiment")
        else:
            st.warning("Please enter some text to analyze.")

elif choice == "Speech-to-Text":
    st.write("Convert your audio file to text.")
    audio_file = st.file_uploader("Upload Audio File (WAV format recommended)", type=["wav"])
    if st.button("Transcribe"):
        if audio_file is not None:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.write("**Transcription:**")
                    st.success(text)
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand the audio.")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
        else:
            st.warning("Please upload an audio file.")

elif choice == "Text-to-Speech":
    st.write("Convert text to speech audio.")
    text_input = st.text_area("Enter text here:", height=150)
    if st.button("Convert to Speech"):
        if text_input:
            tts = gTTS(text=text_input, lang='en')
            tts.save("output.mp3")
            audio_file = open("output.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning("Please enter some text to convert.")

elif choice == "Emotion Detection":
    st.write("Detect emotions present in your text.")
    text_input = st.text_area("Enter text here:", height=150)
    if st.button("Detect Emotions"):
        if text_input:
            emotion = NRCLex(text_input)
            st.write("**Emotion Frequencies:**")
            st.json(emotion.affect_frequencies)
            st.write("**Top Emotions:**")
            st.write(emotion.top_emotions)
        else:
            st.warning("Please enter some text to analyze.")

elif choice == "Spam Detection":
    st.write("Check if your text/message is spam or not.")
    text_input = st.text_area("Enter text here:", height=150)
    if st.button("Check Spam"):
        if text_input:
            with st.spinner("Analyzing..."):
                spam_detector = load_spam_detector()
                result = spam_detector(text_input)[0]
                label = result['label']
                score = result['score']
                if label.lower() == 'spam' or label.lower() == 'label_1':
                    st.error(f"Spam Detected! (Confidence: {score:.2f})")
                else:
                    st.success(f"Not Spam (Confidence: {score:.2f})")
        else:
            st.warning("Please enter some text to check.")

elif choice == "Text Summarization":
    st.write("Summarize long pieces of text.")
    text_input = st.text_area("Enter text here:", height=200)
    if st.button("Summarize"):
        if text_input:
            if len(text_input.split()) < 30:
                st.warning("Please enter a longer text for summarization.")
            else:
                with st.spinner("Summarizing..."):
                    summarizer = load_summarizer()
                    summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)
                    st.write("**Summary:**")
                    st.success(summary[0]['summary_text'])
        else:
            st.warning("Please enter some text to summarize.")

elif choice == "Grammar Checking":
    st.write("Check and correct grammar in your text.")
    text_input = st.text_area("Enter text here:", height=150)
    if st.button("Check Grammar"):
        if text_input:
            with st.spinner("Checking grammar..."):
                tool = load_grammar_tool()
                matches = tool.check(text_input)
                if not matches:
                    st.success("No grammar issues found!")
                else:
                    st.error(f"Found {len(matches)} issues:")
                    for match in matches:
                        st.write(f"- {match.message}")
                    
                    st.write("**Corrected Text:**")
                    st.success(language_tool_python.utils.correct(text_input, matches))
        else:
            st.warning("Please enter some text to check.")

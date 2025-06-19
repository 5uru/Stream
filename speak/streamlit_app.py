import streamlit as st
import random
import time
from main import transcribe_audio, generate_response, generate_audio
from prompt_managements import pm

st.write("Speak Demo App")
st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")
groq_api_key = st.text_input("Enter your Groq API key", type="password")

# Initialiser le contexte dans session_state s'il n'existe pas déjà
if "context" not in st.session_state:
    st.session_state.context = ""
if "chat" not in st.session_state:
    st.session_state.chat = []
if "voice" not in st.session_state:
    st.session_state.voice = "af_heart"  # Default voice

col1, col2 = st.columns(2, border=True)
with col1:
    st.write("**Context Prompt**")
    situation = st.text_input("Situation", placeholder="Describe the situation")
    context_prompt = pm.get_prompt("context_prompt", variables={"Situation": situation})
    if st.button("Generate Context Prompt"):
        if groq_api_key:
            st.session_state.context = generate_response(context_prompt, groq_api_key)
        else:
            st.error("Please enter your Groq API key to generate the context.")
with col2:
    # Random context generation
    st.write("Generating random context.")
    if st.button("Generate Random Context", use_container_width=True):
        if groq_api_key:
            st.session_state.context = generate_response(pm.get_prompt("random_context"), groq_api_key)
        else:
            st.error("Please enter your Groq API key to generate a random context.")

if st.session_state.context:
    st.write("**Context:**")
    st.info(st.session_state.context)
# Voice choice
st.write("**Voice Selection**")
voices_names = ["American Woman 1", "American Woman 2", "American Man", "British Woman", "British Man"]
voices = ["af_heart", "af_bella", "am_fenrir", "bf_emma", "bm_fable"]  # Corresponding voice codes
voice_choice = st.selectbox("Select a voice", options=voices_names, index=0)
selected_voice = voices[voices_names.index(voice_choice)]
st.session_state.voice = selected_voice
for msg in st.session_state.chat:
    with st.container(border=True):
        if msg["role"] == "user":
            st.write("**Me**")
        else:
            st.write("**Assistant**")
        st.audio(msg["audio"], format="audio/wav")
        with st.expander("Show details", expanded=False):
            st.write(f"**Message:** {msg['content']}")
audio_col, btn_col = st.columns([3, 1])
with audio_col:
    audio_value = st.audio_input("Record a voice message")
with btn_col:
    st.write("")
    st.write("")
    st.write("")
    if st.button("Send", use_container_width=True):
        if not audio_value:
            st.error("Please record a voice message before sending.")
        else:
            audio_bytes = audio_value.read()
            text = transcribe_audio(audio_bytes)
            st.session_state.chat.append({"role": "user", "content": text, "audio": audio_bytes})
            chat_history = "".join(
                    f"{msg['role'].capitalize()}: {msg['content']}\n"
                    for msg in st.session_state.chat
            )
            # Utiliser le contexte stocké dans session_state
            ai_response = generate_response(
                    pm.get_prompt("chat_prompt", variables={"context": st.session_state.context, "ChatHistory": chat_history}),
                    groq_api_key
            )
            audio = generate_audio(ai_response, st.session_state.voice)
            st.session_state.chat.append({"role": "you", "content": ai_response, "audio": audio})
            st.rerun()
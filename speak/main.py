import whisper
import tempfile
import getpass
import os
from kokoro import KPipeline
from langchain.chat_models import init_chat_model
import torchaudio
import io



def transcribe_audio(audio_data):
    try:
        # Créer un fichier temporaire pour stocker les données audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # Charger le modèle Whisper
        model = whisper.load_model("base.en")  # ou toute autre taille de modèle

        # Transcrire l'audio en utilisant le chemin du fichier
        result = model.transcribe(temp_audio_path)

        # Supprimer le fichier temporaire
        os.unlink(temp_audio_path)

        return result["text"]
    except Exception as e:
        return f"Erreur de transcription : {str(e)}"


def generate_response(prompt, groq_api_key):
    # Cette fonction est un espace réservé pour la génération de réponses.
    # Vous pouvez intégrer ici votre modèle de langage préféré.
    os.environ["GROQ_API_KEY"] = groq_api_key
    model = init_chat_model("compound-beta", model_provider="groq",)
    response = model.invoke(prompt)
    return response.content if response else "Aucune réponse générée."

def generate_audio(text, voice):
    text  = text.replace('\n', ' ')  # Nettoyer le texte pour éviter les problèmes de segmentation
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(
            text, voice=voice, # <= change voice here
            speed=1, split_pattern=r'\n+'
    )

    for gs, ps, audio in generator:
        # Convertir le tensor en bytes pour Streamlit
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0), sample_rate=24000, format="wav")
        buffer.seek(0)
        return buffer.read()

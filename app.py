from flask import Flask, request, jsonify
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Groq client initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Function to generate LLM response
def generate_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI mental health assistant. "
                    "Give responses based on the user's emotion and text. "
                    "Keep it clear and concise, and provide steps to help them become positive."
                )
            },
            {"role": "user", "content": user_input}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Text endpoint
@app.route('/text-to-llm', methods=['POST'])
def handle_text():
    data = request.get_json()
    user_input = data.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

# Audio endpoint
@app.route('/audio-to-llm', methods=['POST'])
def handle_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    audio_file = request.files['audio']
    original_path = "temp.3gp"
    wav_path = "converted.wav"
    audio_file.save(original_path)

    try:
        # Convert 3GP to WAV
        audio = AudioSegment.from_file(original_path, format="3gp")
        audio.export(wav_path, format="wav")

        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        response = generate_response(text)
        return jsonify({'transcript': text, 'response': response})

    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError:
        return jsonify({'error': 'Speech recognition service failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

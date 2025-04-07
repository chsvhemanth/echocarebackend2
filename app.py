from flask import Flask, request, jsonify
from groq import Groq
import speech_recognition as sr
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def generate_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI mental health assistant..."},
            {"role": "user", "content": user_input}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

@app.route('/text-to-llm', methods=['POST'])
def handle_text():
    data = request.get_json()
    user_input = data.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

@app.route('/audio-to-llm', methods=['POST'])
def handle_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    audio_file = request.files['audio']
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        response = generate_response(text)
        return jsonify({'transcript': text, 'response': response})
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError:
        return jsonify({'error': 'Speech recognition failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)

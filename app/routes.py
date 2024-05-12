from flask import Blueprint, render_template, request, jsonify
import whisper
from openai import OpenAI
import os

main = Blueprint('main', __name__)
model = whisper.load_model("base")  # You can choose a different model size if necessary

# Configure OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OpenAI API Key set in environment variables")

client = OpenAI(api_key=openai_api_key)


@main.route('/')
def home():
    return render_template('index.html')


@main.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return "No file part", 400
    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filepath = './uploads/' + file.filename
        file.save(filepath)

        # Process audio with Whisper
        audio = whisper.load_audio(filepath)
        audio = whisper.pad_or_trim(audio, 16000 * 180)  # Process first 3 min
        result = model.transcribe(audio)

        # Use OpenAI to generate notes
        system_prompt = "You're helping generate notes for students using lecture transcribed audio recording by whisper. Correct any errors in the transcribed text and organize it in notes format, preferably including math formula if the content is math related: "
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": result['text']
                }
            ]
        )
        notes = response.choices[0].message.content

        return render_template('notes.html', notes=notes)

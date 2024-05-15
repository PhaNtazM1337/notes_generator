from flask import Blueprint, render_template, request, jsonify, send_from_directory
import whisper
from openai import OpenAI
import os
import subprocess

main = Blueprint('main', __name__)
model = whisper.load_model("base")  # You can choose a different model size if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.path.join(current_dir, '..', 'uploads')
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
        system_prompt = f"""Generate well-formatted and organized lecture notes from the following audio transcription. Correct any possible mistakes as transcription may be inaccurate. Make sure you include math equations and not just plain test. Directly output the results."""
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

        # Render template with markdown
        return render_template('notes.html', markdown_content=notes)

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_folder, filename)

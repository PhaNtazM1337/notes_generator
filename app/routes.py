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
        system_prompt = """
        You are a LaTeX expert tasked with formatting lecture notes into a clean, well-organized LaTeX template/document. The content includes mathematical formulas, definitions, and explanations that should be neatly presented using appropriate LaTeX environments like 'equation', 'itemize', and 'enumerate'. Ensure the text is error-free and formatted according to typical academic standards for a mathematical lecture. Here are the transcribed notes:
        """
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

        def extract_latex_from_response(response_text):
            # Define the start and end delimiters
            start_delimiter = "```latex"
            end_delimiter = "```"

            # Find the start of the LaTeX block
            start_index = response_text.find(start_delimiter)
            if start_index == -1:
                return "Error"

            # Adjust the start_index to get the actual starting point of LaTeX code
            start_index += len(start_delimiter)

            # Find the end of the LaTeX block
            end_index = response_text.find(end_delimiter, start_index)
            if end_index == -1:
                return "Error"

            # Extract the LaTeX code
            latex_code = response_text[start_index:end_index].strip()

            return latex_code


        res = extract_latex_from_response(notes)
        if res != "Error":
            notes = res
        tex_filename = os.path.splitext(file.filename)[0] + ".tex"
        tex_filepath = os.path.join('./uploads', tex_filename)
        with open(tex_filepath, 'w', encoding='utf-8') as f:
            f.write(notes)  # Assuming 'notes' is already in LaTeX format

        # Compile LaTeX file to PDF
        pdf_filename = os.path.splitext(file.filename)[0] + ".pdf"
        pdf_filepath = os.path.join('./uploads', pdf_filename)
        subprocess.run(['pdflatex', '-output-directory', './uploads', tex_filepath], check=True)

        # Render template with links to download the PDF and LaTeX files
        return render_template('notes.html', pdf_filename=pdf_filename, tex_filename=tex_filename)

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(upload_folder, filename)

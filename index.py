from flask import Flask, request, jsonify, make_response, render_template_string, session
import requests
import base64
import openai
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = os.environ.get('OPENAI_API_KEY')

# Define the fixed set of colors that can be used in the brush
BRUSH_COLORS = {
    '#f44336': 'red',
    '#ff5800': 'orange',
    '#faab09': 'yellow',
    '#008744': 'green',
    '#0057e7': 'blue',
    '#a200ff': 'purple',
    '#ff00c1': 'pink',
    '#ffffff': 'white',
    '#646765': 'grey',
    '#000000': 'black'
}

@app.route('/proxy')
def proxy_image():
    image_url = request.args.get('url')
    response = requests.get(image_url)
    proxy_response = make_response(response.content)
    proxy_response.headers['Content-Type'] = 'image/jpeg'
    proxy_response.headers['Access-Control-Allow-Origin'] = '*'
    return proxy_response

@app.route('/api/process-drawing', methods=['POST'])
def api_process_drawing():
    try:
        data = request.get_json()
        drawing_data = data['drawing']
        text_description = data['description']

        # Decode image from base64
        image_data = base64.b64decode(drawing_data.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert('RGBA')

        # Extract colors used in the drawing
        raw_colors = {(r, g, b) for r, g, b, a in image.getdata() if a > 0}
        raw_colors_hex = {f"#{r:02x}{g:02x}{b:02x}" for r, g, b in raw_colors}
        used_colors_names = [BRUSH_COLORS[hex_color] for hex_color in raw_colors_hex if hex_color in BRUSH_COLORS]

        # Generate prompt using colors and description
        prompt = generate_prompt(text_description, used_colors_names)
        print(f"Generated prompt for DALL-E: {prompt}")

        # Generate image using the DALL-E API
        image_urls = call_dalle_api(prompt, n=2)
        if not image_urls:
            raise ValueError("Failed to generate images")

        # Generate reappraisal advice text
        reappraisal_text = generate_reappraisal_text(text_description)
        print(f"Generated reappraisal text: {reappraisal_text}")

        return jsonify({'image_urls': image_urls, 'reappraisal_text': reappraisal_text})
    except Exception as e:
        print(f"Error processing drawing: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_prompt(description, colors=None):
    if colors:
        color_description = ', '.join(colors)
        prompt = (
            f"Create a purely visual artistic oil painting drawing using the colors {color_description}, "
            f"that reimagines '{description}' in a positive manner. For example, transforming a gloomy cloud "
            f"into a scene with a rainbow. The image must focus entirely on visual elements without any text, "
            f"letters, or numbers."
        )
    else:
        prompt = (
            f"Create a purely visual artistic oil painting drawing that reimagines '{description}' in a positive manner. "
            f"For example, transforming a gloomy cloud into a scene with a rainbow. The image must focus entirely "
            f"on visual elements without any text, letters, or numbers."
        )
    return prompt

def generate_reappraisal_text(description):
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Generate a short positive cognitive reappraisal advice for a child's description, less than three sentences: {description}",
            max_tokens=80
        )
        if 'choices' in response and len(response.choices) > 0:
            return response.choices[0].text.strip()
        else:
            return "Failed to generate meaningful output. Please refine the prompt."
    except Exception as e:
        print(f"Error generating reappraisal text: {str(e)}")
        return "Could not generate reappraisal text."


def call_dalle_api(prompt, n=2):
    api_key = app.secret_key
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "n": n, "size": "512x512"}

    try:
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        images = response.json().get('data', [])
        if not images:
            print("No images returned from DALL-E.")
        return [image['url'] for image in images]
    except requests.exceptions.RequestException as e:
        print(f"Error from OpenAI API: {e}")
        return []


predefined_sentences = {
    4: "Let's draw. Please use 'Visual Metaphor' on the right.",
    5: "Let's draw. Please use 'Visual Metaphor' on the right.",
    6: "Thank you for participating in the session. You can restart the session if you want to explore more."
}


def generate_art_therapy_question(api_key, question_number, session_history):
    openai.api_key = api_key
    question_prompts = [
        "Generate a question to ask user (children) about their current emotion. Do not use 'kiddo'.",
        "Based on the previous responses, generate a short question for identifying and describing the emotion, such as asking about the intensity of the emotion or where in the body it is felt the most. Users are kids, so please use easy and friendly expressions.",
        "Based on the previous responses, generate a short question that explores the context, such as asking what triggered this emotion or describing the situation or thought that led to these feelings. Users are kids, so please use easy and friendly expressions.",
        "Based on the previous responses, generate a short question that asks the user to describe and visualize their emotion as an 'abstract shape or symbol' to create their own metaphor for their mind. Users are kids, so please use easy and friendly expressions, and provide some metaphors or examples.",
        "Based on the previous responses, generate a short question that asks the user to describe and visualize their emotions as a 'texture' to create their own metaphor for their mind. Users are kids, so please use easy and friendly expressions, and provide some metaphors or examples.",
        "Based on the previous responses, provide personalized cognitive reappraisal advice to help think about the situation that user described in the previous response in a more positive way. Or, if user's previous response was already positive, please assist user to think about the good things they might learn from this experience. Please incorporating a playful and engaging approach consistent with CBT theory. Make sure the advice is directly relevant to the emotions and situations described by the child, using examples or activities that are fun and easy for kids to understand. Also, make this less than three sentences."
    ]
    
    user_responses = " ".join([resp for who, resp in session_history if who == 'You'])
    context = f"Based on the user's previous responses: {user_responses}"

    if 1 <= question_number <= 6:
        prompt_text = f"{context} {question_prompts[question_number - 1]}"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt_text,
            max_tokens=150,
            n=1,
            temperature=0.7
        )
        question_text = response.choices[0].text.strip()

        if question_number in predefined_sentences:
            full_question_text = f"Question {question_number}: {predefined_sentences[question_number]} {question_text}"
        else:
            full_question_text = f"Question {question_number}: {question_text}"

        return full_question_text
    else:
        return "Do you want to restart the session?"




@app.route('/api/question', methods=['POST'])
def api_question():
    data = request.json
    user_response = data.get('response', '')
    session['history'] = session.get('history', [])
    session['responses'] = session.get('responses', [])
    session['question_number'] = session.get('question_number', 1)

    # Store the user's response
    session['history'].append(('You', user_response))
    session['responses'].append(user_response)

    if session['question_number'] < 6:
        question_text = generate_art_therapy_question(
            app.secret_key, session['question_number'], session['history']
        )
        session['history'].append(('Therapist', question_text))
        session['question_number'] += 1
        progress = (session['question_number'] - 1) / 6 * 100
        return jsonify({
            'question': question_text,
            'progress': progress,
            'responses': session['responses'],
            'restart': False
        })
    else:
        # Send all responses back when it's the last question
        all_responses = "\n".join([f"Response {i+1}: {response}" for i, response in enumerate(session['responses'])])
        # Include final advice or feedback if needed
        final_advice = generate_reappraisal_text(session['responses'][-1])
        session.clear()
        return jsonify({
            'question': 'Thank you for participating! Here are all your responses:',
            'progress': 100,
            'responses': all_responses + f"\nFinal Advice: {final_advice}",
            'restart': True
        })


@app.route('/', methods=['GET'])
def home():
    session['history'] = session.get('history', [])
    session['question_number'] = session.get('question_number', 1)
    initial_question = generate_art_therapy_question(
        app.secret_key, session['question_number'], session['history']
    )
    session['history'].append(('Therapist', initial_question))
    session['question_number'] += 1

    latest_question = session['history'][-1][1]
    progress_value = (session['question_number'] - 1) / 6 * 100
    return render_template_string("""
    <html>
        <head>
            <title>Mind Palette for kids!</title>
            <style>
                body {
                    font-family: 'Helvetica', sans-serif;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    display: flex;
                    width: 100%;
                }
                .left, .right {
                    width: 50%;
                    padding: 20px;
                }
                .divider {
                    background-color: black;
                    width: 2px;
                    margin: 0 20px;
                    height: auto;
                }
                .active-tool {
                    background-color: black;
                    color: white;
                }
                .button-style {
                    color: white;
                    background-color: black;
                    padding: 5px 10px;
                    cursor: pointer;
                    border: none;
                    margin-left: 10px;
                    border-radius: 4px; 
                }
                .helper-text {
                    font-size: 18px; /* Set font size to 18px */
                    line-height: 1.6; /* Adjust line height for better readability */
                    color: black; /* Ensure the text is in black color */
                }
                #question {
                    font-size: 18px; /* Increase the font size for better readability */
                    line-height: 1.6; /* Adjust line height to add more space between lines */
                    margin-bottom: 20px; /* Additional margin below the text for spacing */
                    color: black; 
                }
                
                progress {
                    width: 430px;
                    height: 10px;
                    margin-top: 10px;
                    color: #0057e7;
                    background-color: #eee;
                    border-radius: 3px;
                }
                progress::-webkit-progress-bar {
                    background-color: #eee;
                    border-radius: 3px;
                }
                progress::-webkit-progress-value {
                    background-color: #0057e7;
                    border-radius: 3px;
                }

                img {
                    width: 256px;
                    height: 256px;
                    margin: 10px;
                }
                input[type="text"] {
                    width: 600px;
                    height: 40px;
                    font-size: 18px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                .canvas-container {
                    display: flex;
                    align-items: start;
                    margin-bottom: 10px;
                    margin-top: 30px; 
                }
                canvas {
                    background-color: #f3f4f6;
                    border: 2px solid #cccccc;
                    border-radius: 4px;
                    cursor: crosshair;
                }
                .brush {
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    cursor: pointer;
                    display: inline-block;
                    margin: 5px;
                }
                .tool-button {
                    background-color: white;
                    border: 1.5px solid black;
                    color: black;
                    padding: 4px 9px;
                    cursor: pointer;
                    margin-left: 13px;
                    border-radius: 4px;
                }
                .spinner {
                    display: inline-block;
                    vertical-align: middle;
                    border: 4px solid rgba(0,0,0,.1);
                    border-radius: 50%;
                    border-left-color: #09f;
                    animation: spin 1s ease infinite;
                    width: 20px;
                    height: 20px;
                }
                #loading p {
                    display: inline-block;
                    vertical-align: middle;
                    margin: 0;
                    padding-left: 10px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <script>
                function sendResponse() {
                    const response = document.getElementById('response').value;
                    fetch('/api/question', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({'response': response})
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('question').textContent = data.question;
                        document.getElementById('response').value = '';
                        document.querySelector('progress').value = data.progress;
                
                        if (data.progress === 100) {
                            showReflectionArea(data.responses);
                            document.getElementById('reflectionButton').style.display = 'block';
                        } else {
                            document.getElementById('reflectionButton').style.display = 'none';
                        }
                    })
                    .catch(error => console.error('Error:', error));
                    return false;
                }
                
                function showReflectionArea(responses) {
                    const reflectionContainer = document.getElementById('reflectionContainer');
                    reflectionContainer.innerHTML = '';  // Clear any previous reflections
                
                    const responseList = document.createElement('div');
                    responseList.innerHTML = responses.split('\n').map((resp, index) => `<p>${resp}</p>`).join('');
                    reflectionContainer.appendChild(responseList);
                    reflectionContainer.style.display = 'block';
                }
            </script>
        </head>
        <body>
            <div class="container">
                <div class="left">
                    <h1>Mind Palette for kids!</h1>
                    <div id="question">{{ latest_question }}</div>
                    <progress value="{{ progress_value }}" max="100"></progress>
                    <form onsubmit="return sendResponse();">
                        <input type="text" id="response" autocomplete="off" placeholder="Enter your response here..." />
                        <input type="submit" value="Respond" class="button-style" />
                        <button id="reflectionButton" class="button-style" style="display: none;" onclick="viewReflection()">Reflection</button>
                    </form>
                    <div class="canvas-container">
                        <canvas id="drawingCanvas" width="500" height="330"></canvas>
                    </div>
                </div>
                <div class="divider"></div>
                <div class="right">
                    <h1>Visual Metaphor</h1>
                    <form onsubmit="return generateImage(event);">
                        <label for="description" class="helper-text">I'm here to help you express your emotions. Please describe what you drew on the canvas!</label>
                        <input type="text" id="description" autocomplete="off" placeholder="Describe your drawing..." />
                        <input type="submit" value="Generate" class="button-style" />
                    </form>
                    <div id="loading" style="display: none; text-align: center;">
                        <div class="spinner"></div>
                        <p>Loading...</p>
                    </div>
                    <div id="images"></div>
                    <div id="reappraisalText" style="padding: 20px; font-size: 18px; line-height: 1.6;"></div>
                    <div id="reflectionContainer" style="display: none; margin-top: 20px;"></div>
                </div>
            </div>
        </body>
    </html>
    """, latest_question=latest_question, progress_value=progress_value)

@app.route('/reflection', methods=['GET'])
def reflection():
    responses = session.get('responses', [])
    formatted_responses = "<br>".join([f"Response {i + 1}: {response}" for i, response in enumerate(responses)])
    return render_template_string("""
    <html>
        <head>
            <title>Your Reflections</title>
            <style>
                body { font-family: 'Helvetica', sans-serif; background-color: #f0f8ff; padding: 20px; }
                h1 { color: #333; }
                .responses { margin-top: 20px; background-color: #fff; padding: 20px; border-radius: 5px; }
                .button-style { margin-top: 20px; background-color: #0057e7; color: white; padding: 10px; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <h1>Thank you for reflecting!</h1>
            <div class="responses">{{ responses|safe }}</div>
            <button class="button-style" onclick="window.location.href='/'">Restart Session</button>
        </body>
    </html>
    """, responses=formatted_responses)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

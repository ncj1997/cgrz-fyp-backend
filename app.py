from datetime import datetime
import os
import time
from flask import Flask, jsonify, request, send_file, url_for
import numpy as np
import api.camo_implant as camo_implant  # Import the function from the separate file
import api.pattern_gen as pattern_gen
from PIL import Image

app = Flask(__name__)

TEMP_FOLDER = './temp'

# Route to use the function from the other file
@app.route('/health_check_camo_impan', methods=['GET'])
def health_checker():
    # Call the function from my_functions.py
    result = camo_implant.checkHealth()
    return jsonify({"result": result})

@app.route('/generate-camouflage', methods=['POST'])
def generate_camouflage():
    print("Function Called for Generating Camo")
    if 'images' not in request.files:
        return jsonify({"error": "No images part in the request"}), 400
    
    # Process uploaded images
    image_files = request.files.getlist('images')
    image_list = []



    for img_file in image_files:
        img = Image.open(img_file.stream)
        img = np.array(img)
        image_list.append(img)

    # Extract colors and deep features
    num_colors = int(request.form.get('num_colors', 5))

    final_camo_file_name = pattern_gen.generate_camouflage(image_list=image_list,num_colors=num_colors)

    # timestamp = time.strftime("%Y%m%d-%H%M%S")

     # Return the URL of the saved image to the frontend
    image_url = url_for('static', filename=final_camo_file_name, _external=True)

    return jsonify({'image_url': image_url})

    # return send_file (final_camo_file, mimetype='image/png', as_attachment=True, download_name=f"adaptive_camouflage_{timestamp}.png")


def add_timestamp_to_filename(filename):
    # Extract the file extension
    file_extension = filename.split('.')[-1]
    # Generate the timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add timestamp to the filename
    new_filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.{file_extension}"
    return new_filename

# Define a route for the POST request
@app.route('/apply_camouflage', methods=['POST'])
def apply_camouflage_route():
    if 'original_image' not in request.files or 'camouflage_image' not in request.files:
        return jsonify({'error': 'Missing images'}), 400

    # Get the uploaded files
    original_image = request.files['original_image']
    camouflage_image = request.files['camouflage_image']

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Save the uploaded files with detected formats
    original_image_path = f'{TEMP_FOLDER}/uploaded_original_{timestamp}{original_image.filename}'
    camouflage_image_path = f'{TEMP_FOLDER}/uploaded_camo_{timestamp}{camouflage_image.filename}'

    original_image.save(original_image_path)
    camouflage_image.save(camouflage_image_path)

    # Output path for the camouflaged image
    output_filename = f'camouflaged_output_{timestamp}.jpg'

    # Run the camouflage application
    result_image = camo_implant.apply_camouflage(camouflage_image_path, original_image_path, output_filename)
    # result_image = 1
    # # Return the resulting camouflaged image as a response
    # return send_file(result_image_path, mimetype='image/jpeg')
    if result_image == 1:
        # Return the URL of the saved image to the frontend
        image_url_imprint = url_for('static', filename=f'images/imprint/{output_filename}', _external=True)
        image_url_detections = url_for('static', filename=f'images/detections/{output_filename}', _external=True)

        return jsonify({'detection_image_url': image_url_detections,'imprint_image':image_url_imprint})

    else:
        return jsonify({"error": "No images part in the request"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

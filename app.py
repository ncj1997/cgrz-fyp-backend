from datetime import datetime
import os
import time
from flask import Flask, Response, abort, jsonify, request, send_file, url_for
import numpy as np
import api.camo_implant as camo_implant  # Import the function from the separate file
import api.pattern_gen as pattern_gen
from PIL import Image

from flask_cors import CORS # type: ignore
app = Flask(__name__)
CORS(app)

TEMP_FOLDER = './temp'

# Define the directory where your images are stored
IMAGE_DIRECTORY = 'static/images/patterns'

@app.route('/download-image/<filename>', methods=['GET'])
def download_image(filename):
    try:
        # Build the full path to the file
        file_path = os.path.join(IMAGE_DIRECTORY, filename)

        # Check if the file exists
        if os.path.exists(file_path):
            # Send the file with the proper Content-Disposition header to trigger download
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            abort(404)  # Return a 404 if the file is not found

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if something goes wrong


@app.route('/progress', methods=['POST'])
def progress():
    # Check if the request contains images
    if 'images' not in request.files:
        return Response("No images part in the request", status=400)

    files = request.files.getlist('images')

    # Generator function to stream progress
    def generate():
        try:
            # Step 1: Image Uploading
            yield f"data: Step 1: Images Uploaded \n\n"
            time.sleep(1)  # Simulate the delay for processing

            # Step 2: Image Preprocessing
            yield f"data: Step 2: Image Preprocessing\n\n"
            time.sleep(1)

            # Step 3: Applying Filters
            yield f"data: Step 3: Applying Filters\n\n"
            time.sleep(1)

            # Step 4: Generating Camouflage
            yield f"data: Step 4: Generating Camouflage\n\n"
            time.sleep(1)

            # Step 5: Finishing
            yield f"data: Step 5: Finishing\n\n"
            time.sleep(1)

            # Once processing is done, send the final image URL
            image_url = "http://backend.intelilab.click/static/images/patterns/camouflaged_20240929010354.png"
            yield f"data: Image processed. View at {image_url}\n\n"
        
        except Exception as e:
            yield f"data: Error occurred: {str(e)}\n\n"

    # Return the event-stream response
    return Response(generate(), mimetype='text/event-stream')

# Route to use the function from the other file
@app.route('/health_check', methods=['GET'])
def health_checker():
    print("Function Called for Health Check @ " + time.strftime("%Y%m%d-%H%M%S"))
    return jsonify({"result": "Server Running"})




@app.route('/generate-camouflage', methods=['POST'])
def generate_camouflage():
    print("Function Called for Generating Camo" + time.strftime("%Y%m%d-%H%M%S"))
    if 'images' not in request.files:
        return jsonify({"error": "No images part in the request"}), 400
    
    # Process uploaded images
    image_files = request.files.getlist('images')
    image_list = []



    for img_file in image_files:
        img = Image.open(img_file.stream)
        img = np.array(img)
        image_list.append(img)




    final_camo_file_name = "sample_image_path.jpg"
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

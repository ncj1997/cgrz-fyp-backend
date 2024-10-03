from datetime import datetime
import os
import time
from api.gan_collage import generate_camouflage_and_collage
from api.noise_image_generation import generateNoiseImage
from flask import Flask, Response, abort, jsonify, request, send_file, url_for
import numpy as np
import api.gan_collage as gan_collage  # Import the function from the separate file
# import api.pattern_gen as pattern_gen
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

#_________________________________________________________________

#________________________________________________________________



# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Allowed environment types
ALLOWED_ENV_TYPES = {'forest', 'desert', 'snowy', 'urban'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check if the uploaded file is an image
def is_image_file(file):
    try:
        img = Image.open(file)
        img.verify()  # This will raise an exception if it's not an image
        return True
    except (IOError, SyntaxError):
        return False


@app.route('/generate-camouflage', methods=['POST'])
def generate_camouflage():
    
    print("Function Called for Generating Camo" + time.strftime("%Y%m%d-%H%M%S"))

    env_type = request.form.get('env_type')
    images = request.files.getlist('images')

    # Validate environment type
    if not env_type or env_type not in ALLOWED_ENV_TYPES:
        return jsonify({'error': 'Invalid or missing environment type'}), 400

    # Validate that images are uploaded
    if not images:
        return jsonify({'error': 'No images uploaded'}), 400
    
    timestamp = str(int(time.time()))  # This will create a folder named by the Unix timestamp
    
    # Create a unique folder path for this upload
    folder_path = os.path.join("./temp/", timestamp)
    
    # Make the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    image_list = []

    for image_file in images:
        # Validate the image extension
        if not allowed_file(image_file.filename):
            return jsonify({'error': f'File {image_file.filename} is not an allowed image type'}), 400
        
        # Check if the uploaded file is a valid image
        if not is_image_file(image_file):
            return jsonify({'error': f'File {image_file.filename} is not a valid image'}), 400
        try:
            # Open the uploaded file using Pillow
            img = Image.open(image_file)
            # Convert image to RGB if needed (Pillow might need this for certain formats)
            img = img.convert('RGB')
            # Define the path to save the image
            image_path = os.path.join(folder_path, f"{image_file.filename}.jpg")
            # Save the image as a JPEG (or you can use the original format)
            img.save(image_path, format='JPEG')  # Save as JPEG, but can be PNG or any other
            image_list.append(image_path)
        except Exception as e:
            return jsonify({'error': f"Error saving image {image_file.filename}: {e}"}), 400
    
    base_url = request.host_url

    def generate():

        try:
            # Step 1: Image Uploading
            yield f"data: Step 1: Images Received \n\n"
            time.sleep(1)  # Simulate the delay for processing
            collage_from_GAN = generate_camouflage_and_collage(folder_path,env_type,timestamp)
            # Remove './static/' to get the relative path
            relative_path = collage_from_GAN.replace('./static/', '')
            # Concatenate base_url with relative_path to create full URL
            url_for_gan_collage = f"{base_url}static/{relative_path}"

            # Step 2: 
            yield f"data: Step 2: Passing to Model to generate GAN Pattern. image_url: {url_for_gan_collage} \n\n "
            time.sleep(5)
            #call the collage function here

            # Step 3: 
            noise_blend_image = generateNoiseImage(folder_id=timestamp, existing_image_path=collage_from_GAN)
            yield f"data: Step 3: Generate Noise Blended Image. image_url: {noise_blend_image}\n\n"
            time.sleep(1)

            # Step 4:
            yield f"data: Step 4: Generating Camouflage\n\n"
            time.sleep(1)

            # Step 5: 
            yield f"data: Step 5: Finishing\n\n"
            time.sleep(1)

            # Once processing is done, send the final image URL
            image_url = "http://backend.intelilab.click/static/images/patterns/camouflaged_20240929010354.png"
            yield f"data: Image processed. View at {image_url}\n\n"
        
        except Exception as e:
            yield f"data: Error occurred: {str(e)}\n\n"

    # Return the event-stream response
    return Response(generate(), mimetype='text/event-stream')


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
    result_image = gan_collage.apply_camouflage(camouflage_image_path, original_image_path, output_filename)
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

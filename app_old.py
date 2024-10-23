from datetime import datetime
import os
import time
import cv2
from api.gan_collage import generate_camouflage_and_collage
from api.noise_image_generation import generateNoiseImage
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory, url_for
import numpy as np
import api.gan_collage as gan_collage  # Import the function from the separate file
# import api.pattern_gen as pattern_gen
from PIL import Image

from flask_cors import CORS

from api.yolo_application import SAVE_DIR, check_detection, sse_stream # type: ignore
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
            time.sleep(3)

            # Step 3: Applying Filters
            yield f"data: Step 3: Applying Filters\n\n"
            time.sleep(3)

            # Step 4: Generating Camouflage
            yield f"data: Step 4: Generating Camouflage\n\n"
            time.sleep(3)

            # Step 5: Finishing
            yield f"data: Step 5: Finishing\n\n"
            time.sleep(3)

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
            wait_time = 4  # in seconds, change this value as needed
            steps = [
                {
                    'id': 1,
                    'description': 'Step 1: Collage has been generated.',
                    'imageUrl': 'static/images/premade_images/v2/1.png',
                    'waitTime': 15,
                },
               
                {
                    'id': 2,
                    'description': 'Step 2: Generated Noise Image',
                    'imageUrl': 'static/images/premade_images/v2/2.png',
                    'waitTime': 20,
                },
                {
                    'id': 3,
                    'description': 'Step 3: Analzed dominant colors',
                    'imageUrl': 'static/images/premade_images/v2/3.png',
                    'waitTime': 10,
                },
               
                {
                    'id': 4,
                    'description': 'Step 4: Applied colors to noise image',
                    'imageUrl': 'static/images/premade_images/v2/4.png',
                    'waitTime': 10,
                },
               
                {
                    'id': 5,
                    'description': 'Step 5: First Iteration of Tessellation Completed',
                    'imageUrl': 'static/images/premade_images/v2/5.png',
                    'waitTime': 120,
                },

                {
                    'id': 6,
                    'description': 'Step 6: Second Iteration of Tessellation Completed',
                    'imageUrl': 'static/images/premade_images/v2/6.png',
                    'waitTime': 120,
                },

                {
                    'id': 7,
                    'description': 'Step 7: Final Camouflage Generated',
                    'imageUrl': 'static/images/premade_images/v2/7.png',
                    'waitTime': 110,
                },
               
            ]
            for step in steps:
                time.sleep(step['waitTime'])  # Wait before moving to the next step
                yield f'data: {{"id": {step["id"]}, "description": "{step["description"]}", "imageUrl": "{base_url}{step["imageUrl"]}", "status": "completed"}}\n\n'
            
            

            ##########################################################################
            #                            Step 1: Collage Generation                  #
            ##########################################################################

            # time.sleep(1)  # Simulate the delay for processing
            # collage_from_GAN = generate_camouflage_and_collage(folder_path,env_type,timestamp)
            # # Remove './static/' to get the relative path
            # relative_path = collage_from_GAN.replace('./static/', '')
            # # Concatenate base_url with relative_path to create full URL
            # url_for_gan_collage = f"{base_url}static/{relative_path}"
            # yield f"data: Step 2: Passing to Model to generate GAN Pattern. image_url: {url_for_gan_collage} \n\n "
            # yield f'data: {{"id": 1, "description": "", "imageUrl": "{base_url}{step["imageUrl"]}", "status": "completed"}}\n\n'
            
            # time.sleep(5)




            ##########################################################################
            #                             Step 2: Generated Noise Image              #
            ##########################################################################





            ##########################################################################
            #                             Step 3: Analyzed dominant colors           #
            ##########################################################################





            ##########################################################################
            #                             Step 4: Applied colors to noise image      #
            ##########################################################################




            ##########################################################################
            #            Step 5: First Iteration of Tessellation Completed           #
            ##########################################################################





            ##########################################################################
            #            Step 6: First Iteration of Tessellation Completed           #
            ##########################################################################





            ##########################################################################
            #            Step 7: Second Iteration of Tessellation Completed          #
            ##########################################################################



            # # Step 3: 
            # noise_blend_image = generateNoiseImage(folder_id=timestamp, existing_image_path=collage_from_GAN)
            # yield f"data: Step 3: Generate Noise Blended Image. image_url: {noise_blend_image}\n\n"
            # time.sleep(1)





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


@app.route('/apply_camouflage', methods=['POST'])
def apply_camouflage():


    base_url = request.host_url.rstrip('/') 
    
    # Check if the environment image file is provided
    if 'environment_image' not in request.files:
        return jsonify({'error': 'Environment image file is missing'}), 400
    
    # Check if the camouflage image file is provided
    if 'camouflage_image' not in request.files:
        return jsonify({'error': 'Camouflage image file is missing'}), 400
    
    # Check if the object type is provided in the form data
    if 'object_type' not in request.form:
        return jsonify({'error': 'Object type is missing'}), 400
    # Get the images and object type
    env_image_file = request.files['environment_image']
    camo_image_file = request.files['camouflage_image']
    object_type = request.form['object_type']

    # Object type mapping
    object_types = {
        'human': ['person'],
        'vehicles': ['car', 'bus', 'truck', 'motorcycle']
    }

    if not object_type or object_type not in object_types:
        return jsonify({'error': 'Invalid or missing object type'}), 400
    

    # Load the images
    env_image = cv2.imdecode(np.frombuffer(env_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    camo_image = cv2.imdecode(np.frombuffer(camo_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    

    # Get the list of objects for the specified object type
    selected_objects = object_types.get(object_type, [])

    final_applied_images = sse_stream(env_image, camo_image, selected_objects, base_url)


    detection_result = check_detection(final_applied_images,selected_objects)

        # Use os.path.relpath to get the relative path from the static folder
    relative_path = os.path.relpath(final_applied_images, start='static')
    image_url = f"{base_url}/static/{relative_path.replace(os.sep, '/')}"

    # Return the image URL as JSON
    return jsonify({'image_url': image_url,'detection_result': detection_result})

# Serve the camouflaged image
@app.route('/static/camafalgues/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

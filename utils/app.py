from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io
import base64
import camogen
app = Flask(__name__)
CORS(app)

def get_dominant_colors_from_all(images, num_colors=3):
    all_pixels = []

    for image in images:
        image = image.resize((100, 100))
        image = image.convert("RGB")
        data = np.array(image)
        all_pixels.append(data.reshape((-1, 3)))

    all_pixels = np.vstack(all_pixels)

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(all_pixels)

    colors = kmeans.cluster_centers_
    colors = colors.astype(int)

    hex_colors = ['#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]) for color in colors]

    return hex_colors


@app.route('/upload-files', methods=['POST'])
def upload_images_files():
    if 'images' not in request.files:
        return jsonify({'error': 'No images part in the request'}), 400
    
    files = request.files.getlist('images')
    num_colors = int(request.form.get('num_colors', 3))  # Default to 3 dominant colors if not specified

    if len(files) == 0:
        return jsonify({'error': 'No images selected for uploading'}), 400
    
    images = []
    for file in files:
        try:
            image = Image.open(file.stream)
            images.append(image)
        except Exception as e:
            return jsonify({'error': f"Error processing image {file.filename}: {str(e)}"}), 500

    try:
        dominant_colors = get_dominant_colors_from_all(images, num_colors)
        return jsonify({'dominant_colors': dominant_colors})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_images():
    data = request.get_json()
    images_data_urls = data.get('images')
    num_colors = int(data.get('num_colors', 3))

    if not images_data_urls:
        return jsonify({'error': 'No images provided'}), 400

    images = []

    for data_url in images_data_urls:
        try:
            # Extract base64 part from Data URL
            header, encoded = data_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        except Exception as e:
            return jsonify({'error': f"Error processing image: {str(e)}"}), 500

    try:
        dominant_colors = get_dominant_colors_from_all(images, num_colors)
        return jsonify({'dominant_colors': dominant_colors})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'}), 200

@app.route('/generate-camouflage', methods=['POST'])
def generate_camouflage():
    try:
        # Get JSON data from request
        parameters = request.get_json()

        # Generate camouflage image using camogen library
        image = camogen.generate(parameters)

        # Convert PIL image to binary data to send as response
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

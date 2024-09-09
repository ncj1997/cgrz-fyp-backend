import time
from flask import Flask, jsonify, request, send_file
import api.camo_implant as camo_implant  # Import the function from the separate file
import api.pattern_gen as pattern_gen
from PIL import Image

app = Flask(__name__)

# Route to use the function from the other file
@app.route('/health_check_camo_impan', methods=['GET'])
def health_checker():
    # Call the function from my_functions.py
    result = camo_implant.checkHealth()
    return jsonify({"result": result})

@app.route('/generate-camouflage', methods=['POST'])
def generate_camouflage():
 
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

    final_camo_file = pattern_gen.generate_camouflage(image_list=image_list,num_colors=num_colors)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    return send_file (final_camo_file, mimetype='image/png', as_attachment=True, download_name=f"adaptive_camouflage_{timestamp}.png")


if __name__ == '__main__':
    app.run(debug=True, port=5000)

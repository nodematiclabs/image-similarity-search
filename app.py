from flask import Flask, request, render_template, send_from_directory
from pymilvus import Collection, connections
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

import io

app = Flask(__name__)

# Initialize Milvus connection and ResNet50 model
connections.connect(host='mymilvus', port='19530')
collection = Collection("image_collection")
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

IMAGES = {
    "448273684311577432": "Schloss_Sigmaringen_2022.jpg",
    "448273684311577434": "Canary_Wharf_from_Limehouse_London_June_2016_HDR.jpg",
    "448273684311577436": "Daslook._Allium_ursinum,_zwellende_bloemknop._18-04-2022_(actm.)_04.jpg"
}

@app.route('/', methods=['GET'])
def index():
    # Simple form for file upload
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Load and preprocess the image
        file_bytes = io.BytesIO(file.read())
        img = image.load_img(file_bytes, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)


        # Extract features
        features = model.predict(x)

        # Create a search parameter
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Perform search
        results = collection.search(data=[features.flatten().tolist()], anns_field="embedding", param=search_params, limit=10)
        
        # Get the most similar image
        similar_image_id = str(results[0].ids[0])
        similar_image_filename = IMAGES.get(similar_image_id, "No similar image found")
        
        # Assuming you have a static directory for serving images
        return send_from_directory('static/images', similar_image_filename)

if __name__ == '__main__':
    app.run(debug=True)

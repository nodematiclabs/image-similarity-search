from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

import re

IMAGES = {
    "448273684311577432": "Schloss_Sigmaringen_2022.jpg",
    "448273684311577434": "Canary_Wharf_from_Limehouse_London_June_2016_HDR.jpg",
    "448273684311577436": "Daslook._Allium_ursinum,_zwellende_bloemknop._18-04-2022_(actm.)_04.jpg"
}

connections.connect(host='localhost', port='19530')

collection = Collection("image_collection")

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load an image
img_path = 'Neillia_affinis,_trosspirea._23-05-2022_(actm.).jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extract features
features = model.predict(x)

# Create a search parameter
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# Assume `query_vector` is the feature vector of the image you're searching for
results = collection.search(data=[features.flatten().tolist()], anns_field="embedding", param=search_params, limit=10)

# Get the most similar image filename, based on the Milvus approximate nearest neighbor search
IMAGES[str(results[0].ids[0])]
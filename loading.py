from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_PATHS = [
    'Schloss_Sigmaringen_2022.jpg',
    'Canary_Wharf_from_Limehouse_London_June_2016_HDR.jpg',
    'Daslook._Allium_ursinum,_zwellende_bloemknop._18-04-2022_(actm.)_04.jpg',
]

connections.connect(host='localhost', port='19530')

# Define the fields for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048)
]

# Define the schema for the collection
schema = CollectionSchema(fields, description="Image collection")

# Specify the index parameters
index_params = {
    "index_type": "IVF_FLAT", # Choose the index type suitable for your use case
    "params": {"nlist": 1024}, # Index parameters (vary based on index type)
    "metric_type": "L2" # Distance metric (e.g., L2, IP)
}

# Create the collection in Milvus
collection_name = "image_collection"
collection = Collection(name=collection_name, schema=schema)
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


for img_path in IMG_PATHS:
    # Load an image
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features
    features = model.predict(x)

    # Prepare the data for insertion
    data = [
        {"embedding": features.flatten().tolist()}
    ]

    # Insert the data into the collection
    mr = collection.insert(data)

    # Remember to save the ids returned by Milvus for future reference
    ids = mr.primary_keys

    print(ids, img_path)
import os
from Data_Acc.data_acc import acquire_datasets
import Utility.consts as consts
from Data_Storage.data_storage import push_data,contains_data,read_data
from Preproc.preproc import preprocess_data
from Data_Exp.data_exp import explore
from ML.inference import inference


# Obtain the database and collection names from the contants.py file
db_name = consts.DATABASE_NAME
collections_names = consts.Collections

preprocessed_data = {}
for collection_name, _ in collections_names.__members__.items():
    print(f"Data to be acquired & stored: {collection_name}")

    # acquire the necessary data
    collection_data = acquire_datasets(collection_name)

    # Check if data exists in the database. If not, insert the data into there.
    if not contains_data(db_name, collection_name):
        push_data(collection_data, db_name, collection_name)

    # Collect data from the MongoDB database.
    stored_data = read_data(db_name, collection_name)

    # Perform data processing
    processed_data = preprocess_data(collection_name, stored_data)
    preprocessed_data[collection_name] = processed_data

# Perform data exploration
explore(preprocessed_data)

# Perform data inference
inference(preprocessed_data)

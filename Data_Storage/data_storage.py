"""
This file implements a CRUD interface for the MongoDB Atlas Database.
"""

import json
import os
import pandas as pd
import pymongo
from Utility.consts import MONGODB_SERVER_ADDRESS
from Utility.utils import decrypt


def get_mongo_credentials():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    credential_file = os.path.join(current_folder, "..", "Utility", "mongodb.json")

    # load username, password from file
    with open(credential_file, "r", encoding="utf-8") as file:
        credentials = json.load(file)
        eusername = credentials["eusername"]
        epassword = credentials["epassword"]

        # decrypt username, password
        username = decrypt(eusername)
        password = decrypt(epassword)
    return username, password


def connect_server():
    try:
        username, password = get_mongo_credentials()
        # connect to mongoDB Atlas cluster
        address = MONGODB_SERVER_ADDRESS.format(username=username, password=password)
        server = pymongo.MongoClient(address)
        server.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return server
    except pymongo.errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB client. Error: {e}")
        return ""


def push_data(data, db, col):
    """
    Performs an insert of data into the mongoDB database.

    Args:
        data: The full data as a dict
        db: MongoDB's database name
        col: Collection name
    """
    try:
        # connect to your local mongo instance
        server = connect_server()
        # grab the collection you want to push the data into
        database = server[db]
        collection = database[col]
        # push the data
        return collection.insert_many(data)
    
    except pymongo.errors.OperationFailure as e:
        print(f"Failed to insert data to MongoDB server. Error: {e}")
        return ""


def read_data(db, col, query= None):
    """
    Performs a read of data from the mongoDB database.

    Args:
        db: MongoDB's database name
        col: Collection name
        query: the query to be executed
    """
    try:
        query = query or []
        # connect to your local mongo instance
        server = connect_server()
        # retrieve the data in `collection_name` that matches `query`
        data = server[db][col].aggregate(query)
        return data
    
    except pymongo.errors.OperationFailure as e:
        print(f"Failed to read data from MongoDB server. Error: {e}")
        return ""


def contains_data(db, col):
    """
    This is conducted to prevent duplicated documents.

    Args:
        db: MongoDB's database name
        col: Collection name
    """
    try:
        server = connect_server()
        # check if database exists
        databases = server.list_database_names()
        if db not in databases:
            return False

        # check if collection exists
        collections = server[db].list_collection_names()
        if col not in collections:
            return False

        # check if collection contains elements
        collection = server[db][col]
        if collection.count_documents({}) <= 0:
            return False
        return True
    
    except pymongo.errors.OperationFailure as e:
        print(f"Failed to perform a data entry check within the MongoDB server. Error: {e}")
        return ""
    
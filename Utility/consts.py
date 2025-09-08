"""
This file contains constants like the MongoDB Atlas Database, collections and connection string
"""

from enum import Enum

MONGODB_SERVER_ADDRESS = "mongodb+srv://{username}:{password}@cluster0.r1r0y1f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

DATABASE_NAME = "daps"

Collections = Enum('Collections',('stock','interest','covid'))


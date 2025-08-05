import os
import yaml
import json
from pathlib import Path
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class PromptManager:
    """
    Manages loading, storing, and retrieving prompt templates from a MongoDB database.
    """
    def __init__(self):
        """
        Initializes the PromptManager and connects to MongoDB.
        """
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set. Please create a .env file.")
        
        try:
            self.client = MongoClient(mongo_uri)
            # You can name your database and collection
            self.db = self.client.get_database("taxonomy_db")
            self.collection = self.db.get_collection("prompts")
            # Verify connection
            self.client.admin.command('ping')
            print("PromptManager connected to MongoDB successfully.")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")



    def get_prompt(self, prompt_id: str) -> dict:
        print(f"Fetching prompt '{prompt_id}' from MongoDB...")
        prompt_doc = self.collection.find_one({"prompt_id": prompt_id})
        
        if not prompt_doc:
            raise ValueError(f"Prompt with prompt_id '{prompt_id}' not found in the database.")
            
        return prompt_doc



    def seed_from_directory(self, directory_path: str):
        """
        Loads all prompts from a directory of .yaml/.json files and upserts them
        into the MongoDB collection. This is used to populate or update the database.
        """
        prompt_files = list(Path(directory_path).glob("*.json")) + list(Path(directory_path).glob("*.yaml"))
        
        if not prompt_files:
            print(f"No prompt template files found in '{directory_path}'.")
            return

        print(f"Found {len(prompt_files)} prompt files. Seeding database...")
        
        operations = []
        for file_path in prompt_files:
            try:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                if "prompt_id" not in data:
                    print(f"Skipping file {file_path.name}: missing 'prompt_id'.")
                    continue
                
                # Prepare an upsert operation
                operations.append(
                    UpdateOne(
                        {"prompt_id": data["prompt_id"]},
                        {"$set": data},
                        upsert=True
                    )
                )
            except Exception as e:
                print(f"Error processing file {file_path.name}: {e}")

        if not operations:
            print("No valid prompts to seed.")
            return

        # Execute all operations in a single batch for efficiency
        result = self.collection.bulk_write(operations)
        print(f"Database seeding complete. Upserted {result.upserted_count} new prompts and modified {result.modified_count} existing prompts.")
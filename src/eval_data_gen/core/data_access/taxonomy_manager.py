import os
import yaml
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class TaxonomyManager:
    """
    Manages loading, storing, and retrieving taxonomies from a MongoDB database.
    """
    def __init__(self):
        """
        Initializes the TaxonomyManager and connects to MongoDB.
        """
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set. Please create a .env file.")
        
        # Establish connection to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database("taxonomy_db")
        self.collection = self.db.get_collection("taxonomies")

    def _flatten_and_prepare(self, domain: str, cat_name: str, item: dict) -> dict:
        """
        Flattens a taxonomy item and prepares it for MongoDB insertion.
        This reuses the logic from your original taxonomy_loader.
        """
        # Create a clean, unique ID for the leaf
        topic_cleaned = item['topic'].split(' – ')[0].replace(' ', '').replace('§', '')
        leaf_id = f"{domain}.{cat_name.replace(' ', '')}.{topic_cleaned}"
        
        return {
            "domain": domain,
            "category": cat_name,
            "topic": item["topic"],
            "difficulty": item["difficulty"],
            "leaf_id": leaf_id
        } 

    def load_from_files_and_store(self, taxonomy_dir: str):
        """
        Loads taxonomies from local YAML/JSON files and stores them in MongoDB.
        This is used to seed the database. It uses 'upsert' to avoid duplicates.
        """
        # Support both .yaml and .json files
        taxonomy_paths = list(Path(taxonomy_dir).glob("*.yaml")) + list(Path(taxonomy_dir).glob("*.json"))
        
        if not taxonomy_paths:
            print(f"No taxonomy files found in directory: {taxonomy_dir}")
            return

        print(f"Found {len(taxonomy_paths)} taxonomy files. Seeding database...")
        count = 0
        for path in taxonomy_paths:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(path.read_text(encoding='utf-8'))
            else: # .json
                import json
                data = json.loads(path.read_text(encoding='utf-8'))

            domain = data["domain"]
            for category, items in data["categories"].items():
                for item in items:
                    taxonomy_doc = self._flatten_and_prepare(domain, category, item)
                    # Use update_one with upsert=True to insert if new, or update if exists.
                    # This prevents creating duplicate documents on subsequent runs.
                    self.collection.update_one(
                        {"leaf_id": taxonomy_doc["leaf_id"]},
                        {"$set": taxonomy_doc},
                        upsert=True
                    )
                    count += 1
        print(f"Database seeding complete. Processed {count} taxonomy leaves.")

    def get_all_leaves(self) -> list[dict]:
        """
        Retrieves all taxonomy leaves from MongoDB and formats them for the pipeline.
        """
        print("Fetching all taxonomy leaves from MongoDB...")
        leaves = []
        for doc in self.collection.find({}):
            # Format the document to match the structure the BundleBuilder expects
            leaves.append({
                "id": doc["leaf_id"],
                "label": doc["topic"],
                "difficulty": doc["difficulty"],
            })
        return leaves


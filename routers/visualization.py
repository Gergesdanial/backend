# visualization.py
from fastapi import APIRouter, HTTPException
import pymongo
import os
from dotenv import load_dotenv
from models.models import Project  # Import the Project model

load_dotenv()

# MongoDB setup
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace("{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)
mongodb = pymongo.MongoClient(CONNECTION_URI)
mongodb = mongodb["annotations"]

router = APIRouter()

@router.get("/projects/{projectId}/jobs/{jobId}/tag-frequencies")
async def get_tag_frequencies(projectId: str, jobId: str):
    try:
        # Get the project and job details
        db = mongodb  # Use the MongoDB connection established globally
        # Fetch the project to get its jobs
        project = await Project.get(id=projectId)
        jobs = await project.get_jobs()  # Fetch all jobs related to the project (no need to await here)

        # Find the job by ID
        job = next((j for j in jobs if str(j.id) == str(jobId)), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Access the collection for the given job
        collection_name = job.annotation_collection_name
        collection = db[collection_name]

        # Fetch all annotations for the job
        documents = list(collection.find({}))

        if not documents:
            return {"message": "No annotations found for the given job"}

        # Calculate tag frequencies based on the job type
        tag_frequencies = {}
        if job.type == "text_classification":
            # Text classification jobs store classes differently
            for document in documents:
                # Check if annotations are present in the document
                annotations = document.get("annotations", [])
                if not annotations:
                    continue

                # Extract the classification from the annotations
                classification = annotations[0].get("classes")
                if classification:
                    tag_frequencies[classification] = tag_frequencies.get(classification, 0) + 1
        else:
            # NER or POS jobs: calculate frequencies based on tags
            for document in documents:
                # Check if annotations are present in the document
                annotations = document.get("annotations", [])
                if not annotations:
                    continue

                # Extract tags from the annotations
                tags = annotations[0].get("tags", [])
                for tag_entry in tags:
                    tag = tag_entry.get("tag")
                    if tag:
                        tag_frequencies[tag] = tag_frequencies.get(tag, 0) + 1

        # Return the tag frequencies
        return {
            "message": "Tag frequencies retrieved successfully",
            "status": "success",
            "tag_frequencies": tag_frequencies
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{projectId}/jobs/{jobId}/heatmap-data")
async def get_heatmap_data(projectId: str, jobId: str):
    try:
        # Get the project and job details
        db = mongodb
        project = await Project.get(id=projectId)
        jobs = await project.get_jobs()
        job = next((j for j in jobs if str(j.id) == str(jobId)), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Access the collection for the given job
        collection_name = job.annotation_collection_name
        collection = db[collection_name]

        # Fetch all annotations for the job
        documents = list(collection.find({}))

        if not documents:
            print("No documents found for the given job.")
            return {"message": "No annotations found for the given job"}

        # Create a list to store annotation density per sentence
        annotation_density = []

        # Calculate annotation density for each sentence (each "document" represents a sentence)
        for i, document in enumerate(documents):
            annotations = document.get("annotations", [])
            total_annotations = sum(len(annotation.get("tags", [])) for annotation in annotations)
            annotation_density.append(total_annotations)

            # Debug: Print the total number of annotations for this sentence
            print(f"Sentence {i} Total Annotations: {total_annotations}")

        # Return the annotation density data per sentence
        return {
            "message": "Annotation density data per sentence retrieved successfully",
            "status": "success",
            "annotationDensity": annotation_density
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug: Log the error
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/projects/{projectId}/jobs/{jobId}/matrix-data")
async def get_entity_cooccurrence_matrix(projectId: str, jobId: str):
    try:
        # Connect to MongoDB
        db = mongodb
        project = await Project.get(id=projectId)
        jobs = await project.get_jobs()
        job = next((j for j in jobs if str(j.id) == str(jobId)), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Access the collection for the given job
        collection_name = job.annotation_collection_name
        collection = db[collection_name]

        # Fetch all annotations for the job
        documents = list(collection.find({}))

        if not documents:
            return {"message": "No annotations found for the given job"}

        # Create a dictionary to hold entity co-occurrence counts
        entity_pairs = {}
        entity_set = set()

        # Calculate co-occurrences within each document
        for document in documents:
            annotations = document.get("annotations", [])
            entities_in_doc = {tag_entry.get("tag") for annotation in annotations for tag_entry in annotation.get("tags", [])}
            entities_in_doc.discard(None)

            # Track all entities found for matrix labels
            entity_set.update(entities_in_doc)

            # Update pair counts for all unique pairs of entities in this document
            for entity_a in entities_in_doc:
                for entity_b in entities_in_doc:
                    if entity_a != entity_b:
                        pair = tuple(sorted([entity_a, entity_b]))
                        entity_pairs[pair] = entity_pairs.get(pair, 0) + 1

        # Prepare labels and matrix for the frontend
        labels = sorted(entity_set)
        co_occurrence_matrix = [[entity_pairs.get((a, b), 0) for b in labels] for a in labels]

        return {
            "message": "Entity co-occurrence matrix retrieved successfully",
            "status": "success",
            "matrixData": {
                "labels": labels,
                "co_occurrence": co_occurrence_matrix
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{projectId}/jobs/{jobId}/scatter-data")
async def get_word_scatter_data(projectId: str, jobId: str):
    try:
        # Connect to MongoDB
        db = mongodb
        project = await Project.get(id=projectId)
        jobs = await project.get_jobs()
        job = next((j for j in jobs if str(j.id) == str(jobId)), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Access the collection for the given job
        collection_name = job.annotation_collection_name
        collection = db[collection_name]

        # Fetch all annotations for the job
        documents = list(collection.find({}))

        if not documents:
            return {"message": "No annotations found for the given job"}

        # Dictionary to store word frequency and length by tag type
        word_data = {}
        
        # Process each document to count word frequency and calculate average word length
        for document in documents:
            annotations = document.get("annotations", [])
            for annotation in annotations:
                tags = annotation.get("tags", [])
                for tag_entry in tags:
                    tag_type = tag_entry.get("tag")  # e.g., LOC, ORG, PERS
                    word = tag_entry.get("text", "").strip()
                    word_length = len(word)  # Length in characters

                    if word:
                        if word not in word_data:
                            word_data[word] = {"frequency": 0, "total_length": 0, "tag_type": tag_type}

                        word_data[word]["frequency"] += 1
                        word_data[word]["total_length"] += word_length
                        word_data[word]["tag_type"] = tag_type  # Grouping by tag type

        # Prepare data for the scatterplot
        scatter_data = [
            {
                "word": word,
                "frequency": info["frequency"],
                "average_length": info["total_length"] / info["frequency"],
                "tag_type": info["tag_type"]
            }
            for word, info in word_data.items()
        ]

        return {
            "message": "Scatter data retrieved successfully",
            "status": "success",
            "scatterData": scatter_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








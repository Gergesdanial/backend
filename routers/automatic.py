import uuid
from fastapi import APIRouter, HTTPException, Request
from models.models import NamedEntityRecognitionJob, PartOfSpeechJob, Project, FileDataSource, TextClassificationJob, User
from ML_models.ner import load_ner_model, annotate_text
from ML_models.pos_model import load_pos_model, tag_pos
from ML_models.tc_model import load_tc_model, classify_text
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import json
import re
from datetime import datetime
import pymongo  # Ensure pymongo is imported if not already
from dotenv import load_dotenv  # To use environment variables
import os
from typing import List

router = APIRouter()

# MongoDB setup
mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")  # Update with your MongoDB connection string

# Load models
tc_model = load_tc_model()

# Define the request body model
class TextClassificationRequest(BaseModel):
    classes: list[str]
    job_title: str  

class JobTitleRequest(BaseModel):
    job_title: str
    model: str  # Selected NER model
    custom_tags: List[str]
# Load environment variables if not loaded globally already
load_dotenv()

# Ensure MongoDB connection setup
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace("{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)

# MongoDB Client
mongodb = pymongo.MongoClient(CONNECTION_URI)
mongodb = mongodb["annotations"]  # Assuming "annotations" is your database









# Named Entity Recognition route
@router.post("/NER-annotate/{projectId}/{dataSourceId}")
async def auto_annotate(projectId: str, dataSourceId: str, request: Request, body: JobTitleRequest):
    model_name = body.model
    custom_tags = body.custom_tags

    try:
        ner_model = load_ner_model(model_name)
        project = await Project.get(id=projectId)
        data_source = await FileDataSource.get(id=dataSourceId)
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Override user email with "automatic@annotation" while keeping other details
        user_info = {
            'id': str(user.id),
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': "automatic@annotation",
            'can_add_data': user.can_add_data,
            'can_create_jobs': user.can_create_jobs
        }

        job_title = body.job_title
        annotation_collection_name = f"{job_title}-{uuid.uuid4()}-{data_source.file_name}"

        created_job = await NamedEntityRecognitionJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            active_learning=False,
            tags_list_as_string=str(custom_tags),
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        db = mongodb
        ner_collection = db[created_job.annotation_collection_name]

        # Annotate the text using the NER model
        annotated_data = [annotate_text(data['text'], ner_model) for data in text_data]

        # Save annotated data with modified user info
        for i, (data, (tokens, annotations)) in enumerate(zip(text_data, annotated_data)):
            annotation_record = {
                '_id': i,  # Ensure IDs are sequential for proper ordering
                'data': {
                    'pageTitle': data.get('pageTitle', ''),
                    'language': data.get('language', 'ar'),
                    'text': data['text']
                },
                'fieldToAnnotate': 'text',
                'tags': custom_tags,
                'annotations': [
                    {
                        'user': user_info,
                        'tags': [{'text': token, 'tag': tag} for token, tag in zip(tokens, annotations)]
                    }
                ],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }

            ner_collection.insert_one(annotation_record)

        # Return sorted data by '_id' in ascending order
        sorted_documents = list(ner_collection.find({}).sort('_id', pymongo.ASCENDING))
        
        return {
            "message": f"NER job created and annotated successfully with custom tags using the '{model_name}' model.",
            "status": "success",
            "model_used": model_name,
            "sorted_documents": sorted_documents  # Include sorted documents in the response if needed
        }
    except Exception as e:
        return {
            "message": "An error occurred during annotation",
            "status": "error",
            "detail": str(e)
        }






# Auto-tag POS route
@router.post("/POS-annotate/{projectId}/{dataSourceId}")
async def auto_tag_pos(projectId: str, dataSourceId: str, request: Request, body: JobTitleRequest):
    model_name = body.model
    custom_tags = body.custom_tags
    
    if not custom_tags:
        return {"message": "Custom tags are required.", "status": "error"}
    
    try:
        pos_model = load_pos_model(model_name)
        project = await Project.get(id=projectId)
        data_source = await FileDataSource.get(id=dataSourceId)
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Override user email with "automatic@annotation" while keeping other details
        user_info = {
            'id': str(user.id),
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': "automatic@annotation",
            'can_add_data': user.can_add_data,
            'can_create_jobs': user.can_create_jobs
        }

        job_title = body.job_title
        annotation_collection_name = f"{job_title}-{uuid.uuid4()}-{data_source.file_name}"

        created_job = await PartOfSpeechJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            active_learning=False,
            tags_list_as_string=str(custom_tags),
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        db = mongodb
        pos_collection = db[created_job.annotation_collection_name]

        # Annotate the text using the POS model
        tagged_data = [tag_pos(data['text'], pos_model) for data in text_data]

        # Save annotated data with modified user info
        for i, (data, (tokens, pos_tags)) in enumerate(zip(text_data, tagged_data)):
            annotation_record = {
                '_id': i,
                'data': {
                    'pageTitle': data.get('pageTitle', ''),
                    'language': data.get('language', 'ar'),
                    'text': data['text']
                },
                'fieldToAnnotate': 'text',
                'tags': custom_tags,
                'annotations': [
                    {
                        'user': user_info,
                        'tags': [{'text': token, 'tag': tag} for token, tag in zip(tokens, pos_tags)]
                    }
                ],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }

            pos_collection.insert_one(annotation_record)

        return {
            "message": f"POS job created and annotated successfully with custom tags using the '{model_name}' model.",
            "status": "success",
            "model_used": model_name
        }
    except Exception as e:
        return {
            "message": "An error occurred during annotation",
            "status": "error",
            "detail": str(e)
        }



# Text classification route
@router.post("/text-classification/{projectId}/{dataSourceId}")
async def get_text_classification(projectId: str, dataSourceId: str, request: Request, body: TextClassificationRequest):
    try:
        tc_model = load_tc_model()
        project = await Project.get(id=projectId)
        data_source = await FileDataSource.get(id=dataSourceId)
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Override user email with "automatic@annotation" while keeping other details
        user_info = {
            'id': str(user.id),
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': "automatic@annotation",
            'can_add_data': user.can_add_data,
            'can_create_jobs': user.can_create_jobs
        }

        job_title = body.job_title
        annotation_collection_name = f"{body.job_title}-{uuid.uuid4()}-{data_source.file_name}"

        created_job = await TextClassificationJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            classes_list_as_string=str(body.classes),
            allow_multi_classification=False,
            active_learning=False,
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        db = mongodb
        tc_collection = db[created_job.annotation_collection_name]

        def classify_text(sentences, classes, model):
            classified_sentences = []
            for sentence in sentences:
                result = model(sentence, classes)
                predicted_class = result['labels'][0]
                classified_sentences.append({"sentence": sentence, "class": predicted_class})
            return classified_sentences

        for i, data in enumerate(text_data):
            sentences = [data['text']]
            classified_sentences = classify_text(sentences, body.classes, tc_model)
            classified_annotations = [{"text": result["sentence"], "class": result["class"]} for result in classified_sentences]

            if classified_annotations:
                annotation_record = {
                    '_id': i,
                    'data': {
                        'fieldToAnnotate': 'text',
                        'text': data['text']
                    },
                    'classes': body.classes,
                    'allowMultiClassification': False,
                    'annotations': [
                        {
                            'user': user_info,
                            'classes': classified_annotations[0]['class']
                        }
                    ],
                    'createdAt': datetime.utcnow(),
                }

                tc_collection.insert_one(annotation_record)

        return {"message": "Text classification job created and annotated successfully", "status": "success"}

    except Exception as e:
        print(f"Error during text classification: {e}")
        return {
            "message": "An error occurred during classification",
            "status": "error",
            "detail": str(e)
        }





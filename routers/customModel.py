from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
from datetime import datetime
from dotenv import load_dotenv
import os
import json
from models.models import Project, FileDataSource, NamedEntityRecognitionJob, PartOfSpeechJob, TextClassificationJob, User
from fastapi import UploadFile, File
import shutil
import textwrap



router = APIRouter()


# Load environment variables
load_dotenv()

# MongoDB setup
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace("{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)

# MongoDB Client
mongodb_client = pymongo.MongoClient(CONNECTION_URI)
custom_models_db = mongodb_client["custom_models_db"]  # Database for custom models
annotations_db = mongodb_client["annotations"]  # Use the original "annotations" database for storing results

# Define the request body model for uploading a custom model
class CustomModelCodeRequest(BaseModel):
    model_name: str  # Name for the uploaded model
    model_code: str  # The literal Python code as a string


# Ensure the customizedModels folder exists
CUSTOM_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "customizedModels")
if not os.path.exists(CUSTOM_MODELS_DIR):
    os.makedirs(CUSTOM_MODELS_DIR)

class CustomNERRequest(BaseModel):
    job_title: str
    custom_tags: List[str]  # Custom tags provided by the user

class JobTitleRequest(BaseModel):
    job_title: str
    model: str  # Selected POS model
    custom_tags: List[str] 


# Endpoint to upload custom NER model code
@router.post("/upload-custom-ner-model-code")
async def upload_custom_ner_model_code(request: Request, body: CustomModelCodeRequest):
    try:
        # Fetch current user ID from the request
        user_id = request.state.user_id

        # Print the code for debugging
        print("Received model code:\n", body.model_code)

        # Prepare the model document
        model_document = {
            "user_id": user_id,
            "model_id": str(uuid.uuid4()),  # Generate a unique model ID
            "model_name": body.model_name,
            "model_code": body.model_code.strip(),  # Strip any extraneous whitespace
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Insert the model document into the MongoDB collection
        models_collection = custom_models_db["custom_models"]
        models_collection.insert_one(model_document)

        # Return a success message
        return {
            "message": "Custom NER model code uploaded successfully.",
            "status": "success",
            "model_id": model_document["model_id"]
        }

    except Exception as e:
        # Handle any exceptions and return an error message
        return {
            "message": "An error occurred while uploading the custom NER model code.",
            "status": "error",
            "detail": str(e)
        }

def load_model_from_code(model_code: str):
    local_scope = {}
    try:
        # Normalize indentation and print code for debugging
        model_code = textwrap.dedent(model_code).strip()
        print("Executing model code:\n", model_code)

        # Check for required function definitions in the code
        if "def load_model(" not in model_code or "def annotate_text(" not in model_code:
            raise ValueError("Model code must define 'load_model' and 'annotate_text' functions.")

        # Execute the uploaded model code
        exec(model_code, {}, local_scope)

        # Validate that 'load_model' and 'annotate_text' functions are in the scope
        if 'load_model' not in local_scope or 'annotate_text' not in local_scope:
            raise ValueError("The provided code does not define 'load_model' or 'annotate_text' functions.")

        # Load the model using the 'load_model' function
        model = local_scope['load_model']()
        return model, local_scope['annotate_text']

    except SyntaxError as e:
        # Handle syntax errors in the uploaded code
        print(f"Syntax error in model code: {e}")
        return None, None
    except Exception as e:
        # Handle all other exceptions
        print(f"Error loading model from code: {e}")
        return None, None


@router.delete("/delete-user-model/{model_id}")
async def delete_user_model(request: Request, model_id: str):
    user_id = request.state.user_id  # Assuming middleware or dependency that extracts user ID
    models_collection = custom_models_db["custom_models"]
    delete_result = models_collection.delete_one({"user_id": user_id, "model_id": model_id})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Model not found or user not authorized to delete this model")
    return {"message": "Model deleted successfully"}




# List custom models for the user
@router.get("/list-user-models")
async def list_user_models(request: Request):
    user_id = request.state.user_id
    models_collection = custom_models_db["custom_models"]
    user_models = models_collection.find({"user_id": user_id}, {"model_name": 1, "_id": 0})
    model_names = [model['model_name'] for model in user_models]
    return {"models": model_names}



# Annotate using a custom NER model
@router.post("/annotate-with-custom-ner-model/{project_id}/{data_source_id}/{model_name}")
async def annotate_with_custom_ner_model(project_id: str, data_source_id: str, model_name: str, request: Request):
    try:
        # Fetch current user ID from the request
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Fetch project and data source
        project = await Project.get(id=project_id)
        data_source = await FileDataSource.get(id=data_source_id)

        # Parse job title and custom tags from the request body
        body = await request.json()
        job_title = body.get("job_title", "Unnamed Job")
        custom_tags = body.get("custom_tags", [])

        if not custom_tags:
            raise HTTPException(status_code=400, detail="Custom tags are required.")

        # Retrieve the model code from the database using model name
        models_collection = custom_models_db["custom_models"]
        model_info = models_collection.find_one({"model_name": model_name})

        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_id = model_info['model_id']
        model_code = model_info['model_code']

        # Load the model and annotation function from the code
        model, annotate_text = load_model_from_code(model_code)
        if not model or not annotate_text:
            raise HTTPException(status_code=500, detail="Failed to load model or annotation function")

        # Create a new NER job for tracking the annotation
        annotation_collection_name = f"{job_title}-{uuid.uuid4()}-{data_source.file_name}"
        created_job = await NamedEntityRecognitionJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            active_learning=False,
            tags_list_as_string=str(custom_tags),  # Use custom tags here
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        # Read the text data from the data source file
        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        # Get the MongoDB collection for the job
        annotations_collection = annotations_db[created_job.annotation_collection_name]

        # Annotate the text using the custom model
        annotated_data = []
        for i, entry in enumerate(text_data):
            tokens, annotations = annotate_text(entry['text'], model)
            annotation_record = {
                '_id': i,
                'data': {
                    'pageTitle': entry.get('pageTitle', ''),
                    'language': entry.get('language', 'ar'),
                    'text': entry['text']
                },
                'fieldToAnnotate': 'text',
                'tags': custom_tags,  # Store custom tags here
                'annotations': [
                    {
                        'user': {
                            'id': str(user.id),
                            'first_name': user.first_name,
                            'last_name': user.last_name,
                            'email': user.email,
                            'can_add_data': user.can_add_data,
                            'can_create_jobs': user.can_create_jobs
                        },
                        'tags': [{'text': token, 'tag': tag} for token, tag in zip(tokens, annotations)]
                    }
                ],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }
            annotated_data.append(annotation_record)

        annotations_collection.insert_many(annotated_data)

        return {
            "message": "NER job created and annotated successfully with custom tags.",
            "status": "success",
            "job_id": str(created_job.id),
            "tags_used": custom_tags  # Include custom tags in the response
        }

    except Exception as e:
        return {
            "message": "An error occurred during annotation.",
            "status": "error",
            "detail": str(e)
        }




# POS tagging with a custom model
@router.post("/annotate-with-custom-pos-model/{project_id}/{data_source_id}/{model_name}")
async def annotate_with_custom_pos_model(project_id: str, data_source_id: str, model_name: str, request: Request):
    try:
        # Fetch current user ID from the request
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Fetch project and data source
        project = await Project.get(id=project_id)
        data_source = await FileDataSource.get(id=data_source_id)

        # Extract job title and custom tags from request body
        body = await request.json()
        job_title = body.get("job_title", "Unnamed Job")
        custom_tags = body.get("custom_tags", [])

        if not custom_tags:
            raise HTTPException(status_code=400, detail="Custom tags are required.")

        # Retrieve the model code from the database using the model name
        models_collection = custom_models_db["custom_models"]
        model_info = models_collection.find_one({"model_name": model_name})

        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_code = model_info['model_code']

        # Load the model and annotation function from the code
        model, tag_pos = load_model_from_code(model_code)
        if not model or not tag_pos:
            raise HTTPException(status_code=500, detail="Failed to load model or annotation function")

        # Create a new POS job for tracking the annotation
        annotation_collection_name = f"{job_title}-{uuid.uuid4()}-{data_source.file_name}"
        created_job = await PartOfSpeechJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            active_learning=False,
            tags_list_as_string=str(custom_tags),  # Use custom tags here
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        # Read the text data from the data source file
        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        # Get the MongoDB collection for the job
        annotations_collection = annotations_db[created_job.annotation_collection_name]

        # Annotate the text using the custom POS model
        annotated_data = []
        for i, entry in enumerate(text_data):
            tokens, pos_tags = tag_pos(entry['text'], model)
            pos_annotations = [{'text': token, 'tag': tag} for token, tag in zip(tokens, pos_tags)]

            # Create the annotation record
            annotation_record = {
                '_id': i,
                'data': {
                    'pageTitle': entry.get('pageTitle', ''),
                    'language': entry.get('language', 'ar'),
                    'text': entry['text']
                },
                'fieldToAnnotate': 'text',
                'tags': custom_tags,  # Use custom tags
                'annotations': [
                    {
                        'user': {
                            'id': str(user.id),
                            'first_name': user.first_name,
                            'last_name': user.last_name,
                            'email': user.email,
                            'can_add_data': user.can_add_data,
                            'can_create_jobs': user.can_create_jobs
                        },
                        'tags': pos_annotations
                    }
                ],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }
            annotated_data.append(annotation_record)

        # Insert the annotation records
        annotations_collection.insert_many(annotated_data)

        return {
            "message": "POS job created and annotated successfully with custom tags.",
            "status": "success",
            "job_id": str(created_job.id),
            "tags_used": custom_tags
        }

    except Exception as e:
        return {
            "message": "An error occurred during annotation.",
            "status": "error",
            "detail": str(e)
        }






# Text classification with a custom model
@router.post("/annotate-with-custom-tc-model/{project_id}/{data_source_id}/{model_name}")
async def annotate_with_custom_tc_model(project_id: str, data_source_id: str, model_name: str, request: Request):
    try:
        # Fetch current user ID from the request
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Fetch project and data source
        project = await Project.get(id=project_id)
        data_source = await FileDataSource.get(id=data_source_id)

        # Extract job title and classes from the request body
        body = await request.json()
        job_title = body.get("job_title", "Unnamed Job")
        classes = body.get("classes", [])

        # Retrieve the model code from the database using the model name
        models_collection = custom_models_db["custom_models"]
        model_info = models_collection.find_one({"model_name": model_name})

        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        model_code = model_info['model_code']

        # Load the model and annotation function from the code
        model, classify_text = load_model_from_code(model_code)
        if not model or not classify_text:
            raise HTTPException(status_code=500, detail="Failed to load model or classification function")

        # Create a new Text Classification job
        annotation_collection_name = f"{job_title}-{uuid.uuid4()}-{data_source.file_name}"
        created_job = await TextClassificationJob.create(
            title=job_title,
            project=project,
            file_data_source=data_source,
            annotation_collection_name=annotation_collection_name,
            field_to_annotate='text',
            classes_list_as_string=str(classes),
            allow_multi_classification=False,
            active_learning=False,
            created_by=user
        )

        await created_job.assigned_annotators.add(user)

        # Read the text data from the data source file
        with open(data_source.location, 'r') as file:
            text_data = json.load(file)

        # Get the MongoDB collection for the job
        annotations_collection = annotations_db[created_job.annotation_collection_name]

        # Annotate the text using the custom text classification model
        annotated_data = []
        for i, entry in enumerate(text_data):
            classified_sentences = classify_text(entry['text'], classes, model)

            # Create the annotation record
            annotation_record = {
                '_id': i,
                'data': {
                    'fieldToAnnotate': 'text',
                    'text': entry['text']
                },
                'classes': classes,
                'allowMultiClassification': False,
                'annotations': [
                    {
                        'user': {
                            'id': str(user.id),
                            'first_name': user.first_name,
                            'last_name': user.last_name,
                            'email': user.email,
                            'can_add_data': user.can_add_data,
                            'can_create_jobs': user.can_create_jobs
                        },
                        'classes': classified_sentences[0]['class'] if classified_sentences else "Unknown"
                    }
                ],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }
            annotated_data.append(annotation_record)

        # Insert the annotation records
        annotations_collection.insert_many(annotated_data)

        return {
            "message": "Text Classification job created and annotated successfully.",
            "status": "success",
            "job_id": str(created_job.id)
        }

    except Exception as e:
        return {
            "message": "An error occurred during annotation.",
            "status": "error",
            "detail": str(e)
        }




# Endpoint to upload any file
@router.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file in the customizedModels folder
        file_path = os.path.join(CUSTOM_MODELS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": "File uploaded successfully.",
            "status": "success",
            "file_path": file_path
        }

    except Exception as e:
        return {
            "message": "An error occurred while uploading the file.",
            "status": "error",
            "detail": str(e)
        }

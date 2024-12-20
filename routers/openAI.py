# openAI.py

import uuid
from fastapi import APIRouter, HTTPException, Request
from models.models import NamedEntityRecognitionJob, PartOfSpeechJob, TextClassificationJob, Project, FileDataSource, User
from pydantic import BaseModel
from datetime import datetime
import pymongo
import os
import openai
import json
import re
import time


# Environment and MongoDB setup
from dotenv import load_dotenv
load_dotenv()
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace("{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)
mongodb = pymongo.MongoClient(CONNECTION_URI)
mongodb = mongodb["annotations"]

router = APIRouter()



# Request model for annotation API
class AnnotationRequest(BaseModel):
    job_title: str
    model: str
    api_key: str
    tags: list[str]
    type: str  # 'ner', 'pos', or 'text_classification'



def determine_max_tokens(text):
    # Estimate based on text length
    if len(text) < 100:
        return 150
    elif len(text) < 200:
        return 250
    else:
        return 300



async def annotate_with_openai(api_key, model, text, tags, task_type):
    openai.api_key = api_key
    prompt = generate_prompt(text, tags, task_type)
    max_retries = 3
    chunk_size = 50

    def split_text(text, size):
        return [text[i:i + size] for i in range(0, len(text), size)]

    text_chunks = split_text(text, chunk_size) if len(text) > chunk_size else [text]
    cleaned_annotations = []

    for chunk in text_chunks:
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an assistant that annotates text based on provided tags."},
                        {"role": "user", "content": generate_prompt(chunk, tags, task_type)}
                    ],
                    max_tokens=determine_max_tokens(text)
                )

                response_text = response.choices[0].message["content"].strip()
                print("Raw Response Text:", response_text)

                if task_type == "text_classification":
                    # Directly use the response text as the category for text classification
                    cleaned_annotations = response_text.strip()
                else:
                    # For NER and POS, parse the JSON response
                    response_text = re.sub(r'```json|```', '', response_text).strip()
                    response_text = response_text.replace("'", '"')
                    annotations = json.loads(response_text)
                    if not isinstance(annotations, list):
                        raise ValueError("Expected a list of annotations but got a different format.")

                    for item in annotations:
                        if isinstance(item, dict) and 'text' in item and 'tag' in item:
                            if item['tag'] in tags and item['tag'] != "":
                                cleaned_annotations.append({
                                    'text': item['text'],
                                    'tag': item['tag']
                                })

                break  # Break out of retry loop if successful

            except Exception as e:
                print(f"Error on chunk '{chunk[:30]}...': {e}")
                retry_count += 1
                time.sleep(1)
                if retry_count == max_retries:
                    print("Max retries reached. Skipping this chunk.")
                    break

    return cleaned_annotations




def generate_prompt(text, tags, task_type):
    if task_type == "ner":
        prompt = (
            "You are an advanced annotator specializing in Named Entity Recognition (NER) for Arabic text. "
            "Your task is to carefully analyze the following Arabic text and annotate each word or symbol with one of the provided tags: "
            f"{tags}. Use each tag strictly according to its intended entity type. "
            "If a word or symbol matches multiple tags, choose the most specific tag. If it does not match any tag, assign an empty tag.\n\n"
            f"Text: \"{text}\"\n\n"
            "Please return ONLY a JSON array where each item is a dictionary with 'text' (the word or symbol) and 'tag' (the entity label). "
            "The format for each dictionary should be {{'text': 'word', 'tag': 'TAG'}}. "
            "If there is no matching tag, use an empty string for 'tag'.\n\n"
            "Do not add explanations, notes, or extra text. Return ONLY the JSON array."
        )
    elif task_type == "pos":
        prompt = (
            "You are an advanced annotator specializing in Part of Speech (POS) tagging for Arabic text. "
            "Your task is to analyze the following Arabic text carefully and tag each word or symbol with one of the provided POS tags: "
            f"{tags}. Assign each word or symbol a single, specific POS tag. "
            "If no tag fits, leave an empty tag.\n\n"
            f"Text: \"{text}\"\n\n"
            "Please return ONLY a JSON array, where each item is a dictionary with 'text' and 'tag'. "
            "Each dictionary should look like {{'text': 'word', 'tag': 'TAG'}}. "
            "If no matching tag applies, use an empty string for 'tag'.\n\n"
            "Do not add explanations, notes, or extra text. Return ONLY the JSON array."
        )
    elif task_type == "text_classification":
        prompt = (
            "You are an advanced annotator specializing in text classification for Arabic texts. "
            "Your task is to classify the overall sentiment or main topic of the following Arabic text into one of the provided categories: "
            f"{tags}. Select only the single most relevant category that best fits the text.\n\n"
            f"Text: \"{text}\"\n\n"
            "Please return only the category name that best fits the text without any additional formatting."
        )
    else:
        raise ValueError("Unsupported annotation type")
    return prompt



@router.post("/openai-annotate/{projectId}/{dataSourceId}")
async def openai_annotate(projectId: str, dataSourceId: str, request: Request, body: AnnotationRequest):
    try:
        # Fetch project and data source
        project = await Project.get(id=projectId)
        data_source = await FileDataSource.get(id=dataSourceId)
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Use standardized user email for OpenAI-created annotations
        user_info = {
            'id': str(user.id),
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': "automatic@annotation",
            'can_add_data': user.can_add_data,
            'can_create_jobs': user.can_create_jobs
        }

        # Determine job type and create job in the database
        annotation_collection_name = f"{body.job_title}-{uuid.uuid4()}-{data_source.file_name}"
        job_type = body.type.lower()

        if job_type == 'ner':
            job = await NamedEntityRecognitionJob.create(
                title=body.job_title, project=project,
                file_data_source=data_source, annotation_collection_name=annotation_collection_name,
                field_to_annotate='text', tags_list_as_string=str(body.tags), created_by=user
            )
        elif job_type == 'pos':
            job = await PartOfSpeechJob.create(
                title=body.job_title, project=project,
                file_data_source=data_source, annotation_collection_name=annotation_collection_name,
                field_to_annotate='text', tags_list_as_string=str(body.tags), created_by=user
            )
        elif job_type == 'text_classification':
            job = await TextClassificationJob.create(
                title=body.job_title, project=project,
                file_data_source=data_source, annotation_collection_name=annotation_collection_name,
                field_to_annotate='text', classes_list_as_string=str(body.tags), created_by=user
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported annotation type")

        # Add user as annotator for this job
        await job.assigned_annotators.add(user)

        # Read data source and perform annotations using OpenAI
        with open(data_source.location, 'r') as file:
            text_data = json.load(file)
        
        db = mongodb
        collection = db[annotation_collection_name]

        for i, data in enumerate(text_data):
            # Generate annotation using OpenAI API
            annotation_text = await annotate_with_openai(body.api_key, body.model, data['text'], body.tags, job_type)

            # Structure annotation based on job type
            if job_type in ['ner', 'pos']:
                tokens_and_tags = [item for item in annotation_text if item['tag']]  # Filter out empty tags
                annotation_structure = {
                    'user': user_info,
                    'tags': tokens_and_tags
                }
            elif job_type == 'text_classification':
                annotation_structure = {
                    'user': user_info,
                    'classes': annotation_text.strip()  # For text classification, `classes` should be a single category
                }
            else:
                raise HTTPException(status_code=400, detail="Unsupported annotation type")

            # Create annotation record, using 'classes' for text classification and 'tags' for NER/POS
            annotation_record = {
                '_id': i,
                'data': {
                    'pageTitle': data.get('pageTitle', ''),
                    'language': data.get('language', 'ar'),
                    'text': data['text']
                },
                'fieldToAnnotate': 'text',
                'annotations': [annotation_structure],
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }
            
            # Include `tags` only for `ner` and `pos` jobs
            if job_type in ['ner', 'pos']:
                annotation_record['tags'] = body.tags
            elif job_type == 'text_classification':
                annotation_record['classes'] = body.tags

            collection.insert_one(annotation_record)

        return {"message": f"{body.type.capitalize()} job created and annotated successfully using OpenAI API.", "status": "success"}

    except Exception as e:
        print(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import APIRouter
from fastapi import HTTPException, Request
from models.models import Project, TextClassificationJob, FileDataSource, User, PartOfSpeechJob, NamedEntityRecognitionJob
import json
import pymongo
import os
from dotenv import load_dotenv
from tortoise.expressions import Q
import uuid
from pydantic import BaseModel

load_dotenv()

username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace(  # type: ignore
    "{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)  # type: ignore
mongodb = pymongo.MongoClient(CONNECTION_URI)
mongodb = mongodb["annotations"]

router = APIRouter()



@router.get("/projects/{id}/jobs")
async def get_project_jobs(id, request: Request):
    try:
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        # Check if the user is part of the project
        project = await Project.get(id=id).prefetch_related('assigned_users')
        if user not in await project.assigned_users.all():
            raise HTTPException(status_code=403, detail="User is not part of the project")

        # Retrieve jobs assigned to this project
        jobs = await project.get_jobs()
        
        # Filter jobs the user has permissions for (can add more specific checks here if needed)
        return {'jobs': list(jobs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{id}/jobs")
async def create_job(id, request: Request):
    try:
        user_id = request.state.user_id
        user = await User.get(id=user_id)

        project = await Project.get(id=id)
        job_data = await request.json()
        file_data_source = await FileDataSource.get(id=job_data['dataSource']['id'])
        annotation_type = job_data['type']
        annotation_collection_name = f"{job_data['name']}-{uuid.uuid4()}-{file_data_source.file_name}"
        match annotation_type:
            case "textClassification":
                with open(file_data_source.location, 'r') as original_data:
                    json_data = json.load(original_data)
                    collection = mongodb[annotation_collection_name]

                    for index, record in enumerate(json_data):
                        annotation_record = {
                            "_id": index,
                            "data": record,
                            "fieldToAnnotate": job_data['fieldToAnnotate'],
                            "classes": job_data['classes'],
                            "allowMultiClassification": job_data['allowMultiClassification'],
                            "annotations": [],
                        }
                        collection.insert_one(annotation_record)

                created_job = await TextClassificationJob.create(title=job_data['name'],
                                                                 project=project,
                                                                 file_data_source=file_data_source,
                                                                 annotation_collection_name=annotation_collection_name,
                                                                 field_to_annotate=job_data['fieldToAnnotate'],
                                                                 classes_list_as_string=str(
                                                                     job_data['classes']),
                                                                 allow_multi_classification=job_data[
                                                                     'allowMultiClassification'],
                                                                 active_learning=True if job_data.get(
                                                                     'active_learning') else False,
                                                                 created_by=user)
                await created_job.assigned_annotators.add(user)
                return created_job

            case "partOfSpeech":
                with open(file_data_source.location, 'r') as original_data:
                    json_data = json.load(original_data)
                    collection = mongodb[annotation_collection_name]

                    for index, record in enumerate(json_data):
                        annotation_record = {
                            "_id": index,
                            "data": record,
                            "fieldToAnnotate": job_data['fieldToAnnotate'],
                            "tags": job_data['tags'],
                            "annotations": [],
                        }
                        collection.insert_one(annotation_record)

                created_job = await PartOfSpeechJob.create(title=job_data['name'],
                                                           project=project,
                                                           file_data_source=file_data_source,
                                                           annotation_collection_name=annotation_collection_name,
                                                           field_to_annotate=job_data['fieldToAnnotate'],
                                                           active_learning=True if job_data.get(
                    'active_learning') else False,

                    tags_list_as_string=str(
                    job_data['tags']),
                    created_by=user)
                await created_job.assigned_annotators.add(user)
                return created_job

            case "namedEntityRecognition":
                with open(file_data_source.location, 'r') as original_data:
                    json_data = json.load(original_data)
                    collection = mongodb[annotation_collection_name]

                    for index, record in enumerate(json_data):
                        annotation_record = {
                            "_id": index,
                            "data": record,
                            "fieldToAnnotate": job_data['fieldToAnnotate'],
                            "tags": job_data['tags'],
                            "annotations": [],
                        }
                        collection.insert_one(annotation_record)

                created_job = await NamedEntityRecognitionJob.create(title=job_data['name'],
                                                                     project=project,
                                                                     file_data_source=file_data_source,
                                                                     annotation_collection_name=annotation_collection_name,
                                                                     field_to_annotate=job_data['fieldToAnnotate'],
                                                                     active_learning=True if job_data.get(
                                                                     'active_learning') else False,
                                                                     tags_list_as_string=str(
                    job_data['tags']),
                    created_by=user)
                await created_job.assigned_annotators.add(user)
                return created_job

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects/{projectId}/jobs/{jobId}")
async def get_project_job_by_id(projectId, jobId):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        return {'job': job}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/projects/{projectId}/jobs/{jobId}")
async def delete_job(projectId, jobId):
    try:
        project = await Project.get(id=projectId)

        await project.delete_job(id=jobId)
        return {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/projects/{projectId}/jobs/{jobId}/users")
async def get_job_users(projectId, jobId):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        assigned_reviewer = await job.assigned_reviewer
        users = await project.assigned_users.all()
        assigned_annotators = [user.id for user in await job.assigned_annotators]
        users = [{**dict(user),
                  'isAnnotator': user.id in assigned_annotators,
                  'isReviewer': assigned_reviewer and user.id == assigned_reviewer.id
                  } for user in await project.assigned_users.all()]
        return users
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from active_learning.model import calculate_entropy_for_batch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from models.models import Project
import pandas as pd
import pymongo
import os
from dotenv import load_dotenv
import json
import uuid
import copy
from datetime import datetime
from active_learning.model import train_on_item, FeedForwardSentimentClassifier
import torch
import logging
from collections import Counter
from sklearn.metrics import cohen_kappa_score






load_dotenv()
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
CONNECTION_URI = os.getenv("MONGODB_BASE_URI").replace(  # type: ignore
    "{MONGODB_USERNAME}", username).replace("{MONGODB_PASSWORD}", password)  # type: ignore

mongodb = pymongo.MongoClient(CONNECTION_URI)
mongodb = mongodb["annotations"]

router = APIRouter()



@router.get("/projects/{projectId}/jobs/{jobId}/annotations/summary")
async def get_annotation_summary(projectId: str, jobId: str):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]

        # Calculate total number of annotations and finished annotations
        totalRowCount = collection.count_documents({})
        finished_annotations = collection.count_documents(
            {"annotations": {"$exists": True, "$not": {"$size": 0}}}
        )

        return {
            "totalRowCount": totalRowCount,
            "finishedAnnotations": finished_annotations
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.get("/projects/{projectId}/jobs/{jobId}/annotations")
async def get_job_annotations(projectId, jobId, itemsPerPage: int, page: int, onlyShowUnanotatedData: bool):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        starting_line = page * itemsPerPage

        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]

        # Prepare custom filter based on onlyShowUnanotatedData flag
        custom_filter = {
            "$or": [
                {"annotations": {"$exists": True, "$eq": []}},
                {"annotations": {"$exists": False}},
            ]
        } if onlyShowUnanotatedData else {}

        if job.active_learning:
            # Active learning case with sorting and entropy calculation
            finished_annotation_counts = collection.count_documents(
                {"annotations": {"$exists": True, "$eq": []}}
            )
            data_with_losses = []
            for i in range(0, finished_annotation_counts, itemsPerPage):
                data_t = collection.find({"annotations": {"$exists": True, "$eq": []}}).sort("_id", pymongo.ASCENDING).skip(i).limit(itemsPerPage)
                model = FeedForwardSentimentClassifier(
                    len_unique_tokens=1000,
                    embedding_dim=6,
                    max_tokens=50,
                    hidden_layer_1_n=256,
                    out_n=3
                )
                model.load_state_dict(torch.load(
                    '/home/iyadelwy/Work/Bachelor/multi-modal-lab/backend/active_learning/model_params.pt'
                ))

                data_with_losses.append(
                    (i + itemsPerPage, calculate_entropy_for_batch(list(data_t), model))
                )
                if i > (100 * itemsPerPage):
                    break
            maximum_entropy = max(data_with_losses, key=lambda x: x[1])

            # Fetch data sorted by '_id' for maximum entropy location
            data = collection.find(custom_filter).sort("_id", pymongo.ASCENDING).skip(maximum_entropy[0]).limit(itemsPerPage)

        else:
            # Non-active learning case with sorting by '_id' in ascending order
            data = collection.find(custom_filter).sort("_id", pymongo.ASCENDING).skip(starting_line).limit(itemsPerPage)

        # Count finished annotations
        finished_annotations = collection.count_documents(
            {"annotations": {"$exists": True, "$not": {"$size": 0}}}
        )
        finished_annotations_by_user = collection.count_documents(
            {"annotations": {"$exists": True, "$not": {"$size": 0}}, "annotations.user": "admin"}
        )

        data = list(data)
        for item in data:
            fake_item = copy.deepcopy(item)
            annotations = fake_item['annotations']
            for annotation in annotations:
                del annotation['user']

            are_equal = all(d == annotations[0] for d in annotations)
            item['conflict'] = not are_equal

        # Total row count for the first page
        totalRowCount = collection.count_documents({}) if page == 0 else 0

        # Calculate conflict statistics
        all_annotated_data = list(collection.find(
            {"annotations": {"$exists": True, "$not": {"$size": 0}}}
        ))
        count_of_conflicts = sum(
            1 for item in all_annotated_data
            if not all(d == item['annotations'][0] for d in item['annotations'])
        )

        stats = {'type': job.type}
        if all_annotated_data:
            stats['conflict_percentage'] = f'{(count_of_conflicts / len(all_annotated_data) * 100):.2f}'

        # Additional stats for text classification jobs
        if job.type == 'text_classification':
            pipeline = [
                {"$unwind": {"path": "$annotations", "preserveNullAndEmptyArrays": True}},
                {"$unwind": {"path": "$annotations.classes", "preserveNullAndEmptyArrays": True}},
                {"$group": {"_id": "$annotations.classes", "count": {"$sum": 1}}}
            ]
            result = list(collection.aggregate(pipeline))
            stats['result'] = result

        return {
            "entropy": f'{maximum_entropy[1]:.3f}' if job.active_learning else None,
            "data": data,
            "totalRowCount": totalRowCount,
            "finishedAnnotations": finished_annotations,
            "finishedAnnotationsByUser": finished_annotations_by_user,
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.get("/projects/{projectId}/jobs/{jobId}/annotations/export")
async def export_data(projectId, jobId, type: str):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]
        data_query = collection.find()

        random_id = uuid.uuid4()
        temp_file_path = f'annotations/temp/{collection_name}-{random_id}-data.{type}'

        if type == 'ndjson':
            with open(temp_file_path, 'a+', encoding='utf-8') as temp:
                for item in data_query:
                    temp.write(json.dumps(dict(item), ensure_ascii=False, default=str) + '\n')

        elif type == 'csv':
            # Prepare a list of rows with structured columns for each `text` and its corresponding `tag`
            rows = []
            for item in data_query:
                # Extract the main text
                text = item.get('data', {}).get('text', '')

                # Process each tag individually to create a separate row per tag
                if 'annotations' in item and item['annotations']:
                    for annotation in item['annotations']:
                        for tag in annotation.get('tags', []):
                            # Create a row with the `text` and individual `tag`
                            row = {
                                "text": text,
                                "tag": f"{tag['text']}({tag['tag']})"
                            }
                            rows.append(row)
                else:
                    # If no tags, still add the text with an empty tag field
                    row = {
                        "text": text,
                        "tag": ''  # No tag available
                    }
                    rows.append(row)

            # Save the structured data to a CSV file with UTF-8-BOM for Arabic compatibility
            df = pd.DataFrame(rows)
            df.to_csv(temp_file_path, index=False, encoding='utf-8-sig')  # UTF-8 with BOM for compatibility

        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

        return FileResponse(
            temp_file_path,
            headers={"Content-Disposition": f"attachment; filename={collection_name}-data.{type}"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))





@router.get("/projects/{projectId}/jobs/{jobId}/annotations/{_id}")
async def get_single_annotation(projectId, jobId, _id: int):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]
        data = collection.find_one({'_id': _id})

        return data

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))






@router.post("/projects/{projectId}/jobs/{jobId}/annotations")
async def create_annotation(projectId, jobId, data: Request):
    try:
        annotation_data = await data.json()

        # Fetch project and job details
        project = await Project.get(id=projectId).prefetch_related('created_by')
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]

        # Get the project creator's information
        creator = await project.created_by  # Get the creator object
        creator_id = str(creator.id)  # Convert UUID to string for JSON compatibility
        creator_email = creator.email
        creator_first_name = creator.first_name
        creator_last_name = creator.last_name

        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]

        # Determine the annotation key based on job type
        if job.type == 'named_entity_recognition':
            annotation_key = 'tags'
        elif job.type == 'text_classification':
            annotation_key = 'classes'
        else:
            raise HTTPException(status_code=400, detail="Unsupported job type")

        # Original logic for updating or inserting a user annotation
        if annotation_data.get('wasReviewed'):
            update_query = {
                '$set': {
                    'wasReviewed': True
                }
            }
        else:
            update_query = {
                '$set': {
                    'annotations': annotation_data['annotations'],
                    'createdAt': datetime.utcnow(),
                }
            }

        # Update or insert the annotation data
        result = collection.find_one_and_update(
            {'_id': annotation_data['_id']},
            update_query,
            return_document=pymongo.ReturnDocument.AFTER)

        # Automatic Project Creator Annotation Logic
        # Fetch the updated annotations and filter out the creator's annotation
        existing_data = collection.find_one({"_id": annotation_data["_id"]})

        if existing_data:
            # Filter non-creator annotations
            non_creator_annotations = [
                ann[annotation_key] for ann in existing_data["annotations"] if ann["user"]["id"] != creator_id
            ]

            # Check if there are any non-creator annotations before processing
            if non_creator_annotations:
                # Convert each tag/class to a tuple for counting
                annotation_tuples = [
                    tuple((item["text"], item["tag"]) for item in items) if job.type == 'named_entity_recognition' 
                    else (items,) for items in non_creator_annotations  # Treat `classes` as a single item tuple
                ]

                # Determine if there’s a conflict among annotations
                conflict = any(items != annotation_tuples[0] for items in annotation_tuples)
                creator_annotations = []

                # Apply automatic annotation rules
                if len(annotation_tuples) % 2 == 1:
                    # Odd number of annotators: take the majority vote
                    majority_vote = Counter(annotation_tuples).most_common(1)[0][0]
                    creator_annotations = [
                        {"text": text, "tag": tag} for text, tag in majority_vote
                    ] if job.type == 'named_entity_recognition' else majority_vote[0]
                else:
                    # Even number of annotators
                    if not conflict:
                        # No conflict, so set creator's annotation to the agreed tags/classes
                        creator_annotations = [
                            {"text": text, "tag": tag} for text, tag in annotation_tuples[0]
                        ] if job.type == 'named_entity_recognition' else annotation_tuples[0][0]
                    else:
                        # Conflict exists, leave creator's annotation empty
                        creator_annotations = [] if job.type == 'named_entity_recognition' else ""

                # Update or add the project creator's annotation
                creator_annotation = next((ann for ann in existing_data["annotations"] if ann["user"]["id"] == creator_id), None)
                if creator_annotation:
                    creator_annotation[annotation_key] = creator_annotations
                else:
                    # Add a new annotation for the project creator if none exists
                    existing_data["annotations"].append({
                        "user": {
                            "id": creator_id,  # Creator's ID
                            "email": creator_email,
                            "first_name": creator_first_name,
                            "last_name": creator_last_name,
                            "can_add_data": False,  # Same attributes as other users
                            "can_create_jobs": False  # Same attributes as other users
                        },
                        annotation_key: creator_annotations
                    })

                # Update the collection with the modified annotations
                collection.update_one(
                    {"_id": annotation_data["_id"]},
                    {"$set": {"annotations": existing_data["annotations"]}}
                )

        # Original active learning logic (if applicable)
        if job.active_learning:
            model = FeedForwardSentimentClassifier(
                len_unique_tokens=1000,
                embedding_dim=6,
                max_tokens=50,
                hidden_layer_1_n=256,
                out_n=3
            )
            model.load_state_dict(torch.load('/path/to/model_params.pt'))  # Replace with the actual model path
            train_on_item(result, model)

        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))






@router.get("/projects/{projectId}/jobs/{jobId}/visualization")
async def get_visualization_data(projectId, jobId):
    try:
        project = await Project.get(id=projectId)
        job = await project.get_jobs(id=jobId)  # type: ignore
        job = job[0]
        collection_name = job.annotation_collection_name
        collection = mongodb[collection_name]

        pipeline = [
            {"$unwind": "$annotations"},
            {"$group": {"_id": "$annotations.user.email", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "user_email": "$_id", "count": 1}}
        ]
        result = list(collection.aggregate(pipeline))

        finished_annotations = collection.count_documents(
            {"annotations": {"$exists": True, "$not": {"$size": 0}}}
        )
        totalRowCount = collection.count_documents({})

        date_pipeline = [
            {
                "$match": {
                    "createdAt": {"$exists": True}
                }
            },
            {
                "$project": {
                    "year": {"$year": "$createdAt"},
                    "month": {"$month": "$createdAt"},
                    "day": {"$dayOfMonth": "$createdAt"}
                }
            },
            {
                "$group": {
                    "_id": {"year": "$year", "month": "$month", "day": "$day"},
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "year": "$_id.year",
                    "month": "$_id.month",
                    "day": "$_id.day",
                    "count": 1
                }
            },
            {"$sort": {"year": 1, "month": 1, "day": 1}}
        ]

        date_result = list(collection.aggregate(date_pipeline))

        return {
            'total': totalRowCount,
            'total_finished': finished_annotations,
            'by_user': result,
            'by_date': date_result,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))







@router.get("/projects/{projectId}/jobs/{jobId}/is-creator-annotations")
async def check_project_creator_and_get_annotations(projectId: str, jobId: str, request: Request):
    try:
        user_id = str(request.state.user_id)
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in request context.")

        project = await Project.get(id=projectId).prefetch_related('created_by')
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")

        creator_id = str(project.created_by.id)
        is_creator = creator_id == user_id

        if is_creator:
            jobs = await project.get_jobs(id=jobId)
            if not jobs:
                raise HTTPException(status_code=404, detail="Job not found.")
            job = jobs[0]

            collection_name = job.annotation_collection_name
            collection = mongodb[collection_name]
            annotations = list(collection.find({}))

            # Determine field to include based on job type
            field_name = "classes" if job.type == "text_classification" else "tags"

            # Refine the data to include only document_id, email, and the relevant field
            refined_annotations = []
            for annotation in annotations:
                document_id = annotation['_id']
                user_annotations = [
                    {
                        "email": ann['user']['email'],
                        field_name: ann.get(field_name, [])
                    }
                    for ann in annotation.get('annotations', [])
                    if "user" in ann and "email" in ann['user']
                ]
                refined_annotations.append({
                    "document_id": document_id,
                    "annotations": user_annotations
                })

            return {
                "is_creator": is_creator,
                "annotations": refined_annotations
            }
        else:
            return {"is_creator": is_creator, "annotations": []}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")





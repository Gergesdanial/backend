# authentication.py

from fastapi import APIRouter, HTTPException, Request, Security
from fastapi.security import HTTPBearer
import os
import bcrypt
import jwt
import datetime
from dotenv import load_dotenv
from models.models import User
from tortoise.exceptions import IntegrityError, DoesNotExist

# Load environment variables
load_dotenv()
salt = os.getenv("SALT")
jwt_secret = os.getenv("JWT_SECRET")

router = APIRouter()
auth_scheme = HTTPBearer()

# Middleware function to authenticate user by decoding JWT token
async def authenticate_user(request: Request, token: str = Security(auth_scheme)):
    try:
        # Decode the JWT token
        payload = jwt.decode(token.credentials, jwt_secret, algorithms=["HS256"])
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=403, detail="Invalid authentication token.")
        
        # Set user_id in request.state so it’s accessible in your endpoints
        request.state.user_id = user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token.")

# Sign-up endpoint
@router.post("/signup")
async def sign_up(request: Request):
    try:
        body = await request.json()
        hashed_password = bcrypt.hashpw(
            body['password'].encode(), salt.encode()).decode('utf-8')  # type: ignore
        user = await User.create(
            first_name=body['firstName'],
            last_name=body['lastName'],
            email=body['email'],
            password=hashed_password,
        )
        payload = {
            'user_id': str(user.id),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=90)
        }
        token = jwt.encode(payload, jwt_secret, algorithm='HS256')
        return {'access_token': token}
    except IntegrityError as e:
        raise HTTPException(
            status_code=409, detail='User with this email already exists')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Sign-in endpoint
@router.post("/signin")
async def sign_in(request: Request):
    try:
        body = await request.json()
        email = body['email']
        user = await User.get(email=email)
        if bcrypt.checkpw(body['password'].encode(),
                          user.password.encode()):
            payload = {
                'user_id': str(user.id),
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=90)
            }
            token = jwt.encode(payload, jwt_secret, algorithm='HS256')
            return {'access_token': token}
        else:
            raise Exception('incorrect password')
    except DoesNotExist:
        raise HTTPException(status_code=404, detail='user not found')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to get the current user's data
@router.get("/currentuser")
async def get_current_user(request: Request):
    try:
        user_id = request.state.user_id
        user = dict(await User.get(id=user_id))
        del user['password']
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

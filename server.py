import argparse
import json
import os
import shutil
import secrets
import pipeline
import multiprocessing as mp 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile    
from fastapi.responses import FileResponse
from image2text.driver.driver import Engine

app = FastAPI()
engine = None
vocabulary = None
embeddings = json.load(open('./data/BigBangTheory.json'))
embeddings.update(
    json.load(open('./data/HouseOfCards.json'))
)
parser = argparse.ArgumentParser(description='Start a FastAPI server.')
parser.add_argument('folder', type=str, help='folder to store user specific files')
parser.add_argument('--host', type=str, default='localhost', help='the host to bind to')
parser.add_argument('--port', type=int, default=8000, help='the port to bind to')
args = parser.parse_args()

@app.get('/token')
def token():
    """Генерирует токен для авторизации пользователей
    """
    # generate a 16-character random token
    token = secrets.token_hex(16)
    os.makedirs(os.path.join(args.folder, token), exist_ok=True)
    return token

@app.post('/from-video')
def from_video(token: str = Form(), video: UploadFile = File()):
    """
    Args:
        token:
        video:
    Returns:

    """
    user_folder = os.path.join(args.folder, token)
    if not os.path.exists(user_folder):
        raise HTTPException(
            status_code=403, 
            detail="Invalid token. Generate token using GET /token/ first"
        )
    
    stored_file_path = os.path.join(user_folder, video.filename)
    with open(stored_file_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    result_path = pipeline.process_video(
        token, 
        stored_file_path, 
        os.path.join(args.folder, '__temp__'), 
        engine,
        embeddings,
    )
    return FileResponse(result_path)

@app.post("/from-stream")
def stream(token: str, url: str):
    """
    """
    raise HTTPException(404, "Not supported")


if __name__ == '__main__':
    import uvicorn
    import warnings
    warnings.filterwarnings('ignore')
    engine = Engine(2)
    uvicorn.run(app, host=args.host, port=args.port)

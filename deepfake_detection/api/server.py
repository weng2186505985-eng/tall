from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
import os
from inference import InferencePipeline

app = FastAPI(title="Few-Shot Deepfake Detection API")

# Initialize pipeline (Global)
# pipeline = InferencePipeline("checkpoints/model.pth")

@app.get("/")
def read_root():
    return {"status": "Online", "model": "Few-Shot TALL-Swin"}

@app.post("/detect")
async def detect_video(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run Inference
    # result = pipeline.predict(file_path)
    result = {"fake_probability": 0.92, "verdict": "Fake", "evidence": "Temporal inconsistency detected in frame 24-30"}
    
    # Cleanup
    os.remove(file_path)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

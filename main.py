from typing import Union
from fastapi import FastAPI, File, UploadFile
from detector import get_predict_image

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/prediksi")
def read_root(
    file: UploadFile = File(...)
    ):
    # await print(file.read)
    result = get_predict_image(file)
    return result

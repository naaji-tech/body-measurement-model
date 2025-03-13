import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field

from app.service.call_procedures import call_procedures

app = FastAPI()

class UserInput(BaseModel):
    height: float = Field(..., gt=0, description="Height in meters")
    weight: float = Field(..., gt=0, description="Weight in kilograms")
    age: float = Field(..., gt=0, description="Age in years")
    gender: str = Field(..., description="MALE, FEMALE or OTHER")


@app.post("v1/userMeasurements")
async def measure_body(
    image: UploadFile = File(...),
    height: float = Form(...),
    weight: float = Form(...),
    age: float = Form(...),
    gender: str = Form(...),
):
    """
    API URL end point to measure body dimensions from an image
    """
    try:
        # === Input validation ===
        if not image:
            raise HTTPException(status_code=400, detail="No image file provided")

        if (
            height <= 0
            or weight <= 0
            or age <= 0
            or gender not in ("MALE", "FEMALE", "NEUTRAL")
        ):
            raise HTTPException(
                status_code=400, detail="Invalid height, weight, age or gender values"
            )

        return call_procedures(image, height, weight, age, gender)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

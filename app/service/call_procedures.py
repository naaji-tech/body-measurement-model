import shutil
import tempfile
import os

from fastapi import UploadFile

from app.procedures.extract_beta import extract_beta
from app.procedures.calculate_measurements import calculate_measurements


def call_procedures(
    image: UploadFile, height: float, weight: float, age: float, gender: str
) -> dict:
    """
    This function calls the all procedures to calculate body measurements from an image
    """

    # === Save image temporarily ===
    temp_image_path = tempfile.mktemp(suffix=".jpg")
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # === Log header parameters ===
    print(f"Height: {height}, Weight: {weight}, Age: {age}")

    # === Extract the beta parameters ===
    betas = extract_beta(temp_image_path)

    # === Calculate body measurements ===
    measurements = calculate_measurements(height, gender, betas)

    # === Cleanup temp image ===
    os.remove(temp_image_path)

    return measurements 

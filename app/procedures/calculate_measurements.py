import torch
from pprint import pprint
from lib.smpl_anthropometry.measure import MeasureBody
from lib.smpl_anthropometry.measurement_definitions import STANDARD_LABELS

def calculate_measurements(user_height: float, user_gender: str, betas: torch.tensor) -> dict:
    '''
    description:
        - Calculate body measurements from a given height and body shape using the SMPL body model
        - The measurements are normalized to the given height
    parameters:
        - user_height: float
        - user_gender: str
        - betas: torch.tensor
    returns:
        - dict: A dictionary of labeled measurements
    '''
    print("Measuring smpl body model...")
    measurer = MeasureBody('smpl')
    measurer.from_body_model(gender=user_gender, shape=betas)

    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)
    measurer.height_normalize_measurements(user_height)

    print("Measurements")
    pprint(measurer.height_normalized_measurements)

    print("Labeled measurements")
    pprint(measurer.height_normalized_labeled_measurements)
    
    rounded_measurements = {k: round(v, 2) for k, v in measurer.height_normalized_labeled_measurements.items()}
    return rounded_measurements
# from math import pi
from smplx import SMPL
import torch
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load SMPL Model for 3D Body Mesh
smpl_model = SMPL(
    model_path="data/smpl", gender="male", batch_size=1
)

mediapipe_to_smpl_map = {
    0: 0,  # Nose → Pelvis (approximate)
    11: 2,  # Left Shoulder → Left Shoulder
    12: 5,  # Right Shoulder → Right Shoulder
    13: 3,  # Left Elbow → Left Elbow
    14: 6,  # Right Elbow → Right Elbow
    15: 4,  # Left Wrist → Left Wrist
    16: 7,  # Right Wrist → Right Wrist
    23: 8,  # Left Hip → Left Hip
    24: 11,  # Right Hip → Right Hip
    25: 9,  # Left Knee → Left Knee
    26: 12,  # Right Knee → Right Knee
    27: 10,  # Left Ankle → Left Ankle
    28: 13,  # Right Ankle → Right Ankle
}


def interpolate_missing_joints(smpl_keypoints):
    missing_joints = list(set(range(24)) - set(mediapipe_to_smpl_map.values()))

    for joint in missing_joints:
        if joint in [1, 4]:  # Example: Left Hand (estimate from Wrist & Elbow)
            smpl_keypoints[joint] = (smpl_keypoints[3] + smpl_keypoints[4]) / 2
        elif joint in [10, 13]:  # Example: Right Hand (estimate from Wrist & Elbow)
            smpl_keypoints[joint] = (smpl_keypoints[6] + smpl_keypoints[7]) / 2
        elif joint in [14, 15]:  # Neck (approximate between shoulders)
            smpl_keypoints[joint] = (smpl_keypoints[2] + smpl_keypoints[5]) / 2
        else:
            smpl_keypoints[joint] = smpl_keypoints[0]  # Default to pelvis if unknown

    return smpl_keypoints


# Convert MediaPipe 33 keypoints into a SMPL-compatible pose vector.
def mediapipe_to_smpl_pose(mediapipe_keypoints):
    # Convert keypoints to NumPy array
    keypoints = np.array(mediapipe_keypoints)

    # Extract the relevant 24 keypoints using our mapping
    smpl_keypoints = np.zeros((24, 3), dtype=np.float32)
    print(f"\nsmpl_keypoints: {smpl_keypoints}\n")
    for mp_idx, smpl_idx in mediapipe_to_smpl_map.items():
        print(f"\nmp_idx: {mp_idx}, smpl_idx: {smpl_idx}\n")
        smpl_keypoints[smpl_idx] = keypoints[mp_idx][:3]

    print(f"\nsmpl_keypoints: {smpl_keypoints}\n")

    smpl_keypoints = interpolate_missing_joints(smpl_keypoints)

    # Normalize by setting the pelvis (hip center) as the origin
    pelvis = (
        smpl_keypoints[1] + smpl_keypoints[2]
    ) / 2  # Midpoint of left & right hips
    smpl_keypoints -= pelvis  # Translate to pelvis-centered coordinate system

    print(f"\nsmpl_keypoints: {smpl_keypoints}\n")

    # Convert positions to rotation vectors (Axis-Angle format)
    rotations = []
    for i in range(len(smpl_keypoints) - 1):
        vec = smpl_keypoints[i + 1] - smpl_keypoints[i]  # Compute direction
        r = R.from_rotvec(vec)  # Convert to rotation vector
        rotations.append(r.as_rotvec())

    print(f"\nrotations: {rotations}\n")

    # Flatten the rotations to form the 69D body pose vector
    body_pose = np.concatenate(rotations, axis=0, dtype=np.float32)

    print(f"\nbody_pose: {body_pose}\n")

    # Ensure pose vector shape is (1, 69)
    body_pose = np.expand_dims(body_pose[:69], axis=0)

    return body_pose


mp_keypoints = np.array(
    [
        (0.5209169387817383, 0.29812026023864746, -0.6935145258903503),
        (0.5308256149291992, 0.29089444875717163, -0.6756480932235718),
        (0.5363793969154358, 0.2934950888156891, -0.6760534048080444),
        (0.542243480682373, 0.29661908745765686, -0.6760480403900146),
        (0.5170010924339294, 0.2889104187488556, -0.6736640930175781),
        (0.512440025806427, 0.2898067235946655, -0.6741604804992676),
        (0.5075613856315613, 0.2907079756259918, -0.6744177937507629),
        (0.5465335249900818, 0.3103204071521759, -0.5420365929603577),
        (0.49862560629844666, 0.30532336235046387, -0.5329405069351196),
        (0.5299859642982483, 0.32028937339782715, -0.6389104723930359),
        (0.5103158950805664, 0.31654268503189087, -0.6392236351966858),
        (0.611691415309906, 0.38567259907722473, -0.4466671943664551),
        (0.4293709993362427, 0.3872263431549072, -0.40114492177963257),
        (0.7412422299385071, 0.41306066513061523, -0.48188847303390503),
        (0.2968847155570984, 0.41091465950012207, -0.48951980471611023),
        (0.8647202253341675, 0.4323462247848511, -0.5894590616226196),
        (0.16642549633979797, 0.43356698751449585, -0.6588824391365051),
        (0.9034988880157471, 0.4354148507118225, -0.6035891175270081),
        (0.12881571054458618, 0.43549835681915283, -0.6896093487739563),
        (0.910590648651123, 0.4346240758895874, -0.6565273404121399),
        (0.12291955947875977, 0.4350029230117798, -0.7325778603553772),
        (0.8962046504020691, 0.4357447624206543, -0.6140318512916565),
        (0.13894739747047424, 0.4360833764076233, -0.6856629252433777),
        (0.5772510170936584, 0.6406334042549133, -0.005571062210947275),
        (0.4796788990497589, 0.646369218826294, 0.005021575838327408),
        (0.5663961172103882, 0.6916215419769287, -0.5008567571640015),
        (0.48408424854278564, 0.7106850147247314, -0.48409318923950195),
        (0.5610741972923279, 0.8851750493049622, -0.17181549966335297),
        (0.487819641828537, 0.8845615386962891, -0.16621561348438263),
        (0.551069438457489, 0.9039453268051147, -0.14686542749404907),
        (0.4936542510986328, 0.9027844667434692, -0.1394277662038803),
        (0.58521968126297, 0.9548757076263428, -0.3222227096557617),
        (0.4753918945789337, 0.9527162313461304, -0.3135709762573242),
    ]
)


# Define shape (β) and pose (θ) parameters
# Shape parameters control body shape (e.g., height, weight)
# Pose parameters control joint angles
num_betas = 10

# example pose and shape parameters
# betas = torch.tensor(
#     [
#         [
#             0.1613,
#             1.0595,
#             0.8971,
#             0.9009,
#             -0.4889,
#             0.0425,
#             -0.2025,
#             0.3979,
#             0.1361,
#             -0.0938,
#         ]
#     ]
# )
betas = torch.zeros(1, num_betas)
pose = torch.zeros(1, 69)
# pose = mediapipe_to_smpl_pose(mp_keypoints)

print(f"\nbetas shape: {betas.shape}")
print(f"pose shape: {pose.shape}\n")


# pose[0, 0] = -(pi / 2)
# pose[0, 1] = (pi / 2)
# pose[0, 2] = pi / 2


print(f"\nbetas data: {betas}\n")
print(f"\npose data: {pose}\n")

global_orient = torch.zeros(1, 3, dtype=torch.float32)
# pose = torch.tensor(pose, dtype=torch.float32)

# generate the 3D mesh
output = smpl_model(betas=betas, body_pose=pose, global_orient=global_orient)
vertics = output.vertices.detach().cpu().numpy().squeeze()
faces = smpl_model.faces

# create trimesh object
mesh = trimesh.Trimesh(vertices=vertics, faces=faces)

# visualize the mesh
mesh.show()

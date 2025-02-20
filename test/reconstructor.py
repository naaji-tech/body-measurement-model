import numpy as np
from sympy import false
import torch
from smplx import SMPL
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyrender
import trimesh

def process_keypoints(keypoints):
    """
    Process and normalize the keypoints
    Returns: numpy array of shape (N, 3)
    """
    keypoints = np.array(keypoints)
    # Center the keypoints
    center = np.mean(keypoints, axis=0)
    keypoints = keypoints - center
    return keypoints

def create_smpl_model(gender='neutral'):
    """
    Create and return SMPL model
    """
    model = SMPL(
        model_path='data/smpl',  # Update with your SMPL model path
        gender=gender,
        batch_size=1
    )
    return model

def compute_joints_loss(pred_joints, target_joints):
    # Ensure both arrays have the same shape
    min_length = min(pred_joints.shape[0], target_joints.shape[0])
    pred_joints = pred_joints[:min_length]
    target_joints = target_joints[:min_length]
    return np.mean((pred_joints - target_joints) ** 2)

def optimize_smpl_parameters(model, target_joints, height_m, weight_kg):
    """
    Optimize SMPL parameters to match target joints and measurements
    """
    def objective(x):
        # Split parameters
        pose = x[:72]  # 24 joints * 3
        betas = x[72:]  # Shape parameters
        
        # Forward pass through SMPL
        output = model(
            betas=torch.tensor(betas).unsqueeze(0).float(),
            pose=torch.tensor(pose).unsqueeze(0).float()
        )
        
        # Get predicted joints
        pred_joints = output.joints.detach().numpy()[0]
        # Compute joint loss
        joint_loss = compute_joints_loss(pred_joints, target_joints)
        
        # Add height and weight constraints
        vertices = output.vertices[0].detach()
        height = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).norm().item()
        height_loss = (height - height_m) ** 2
        
        # Approximate weight using volume
        vertices = output.vertices[0].detach().numpy()

        # Check if vertices array is not empty
        if vertices.size > 0:
            # Project 3D vertices to 2D (xy plane) and ensure correct format
            vertices_2d = vertices[:, :2].astype(np.float32)  # Keep only x,y coordinates
            # Ensure correct format for OpenCV
            points = np.float32(vertices_2d).reshape(-1, 1, 2)

            hull = cv2.convexHull(points)
            hull_volume = cv2.contourArea(hull)  # Use contour area as a rough approximation of volume
            pred_weight = hull_volume * 1000  # rough approximation
            weight_loss = ((pred_weight - weight_kg) ** 2)
        else:
            weight_loss = float('inf')  # Assign a large loss if vertices array is empty
        
        total_loss = float(joint_loss) + 0.1 * float(height_loss) + 0.1 * float(weight_loss)
        return total_loss
    
    # Initial guess
    x0 = np.zeros(82)  # 72 pose params + 10 shape params
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='BFGS',
        options={'maxiter': 100}
    )
    
    return result.x

def visualize_keypoints_3d(keypoints, title="3D Keypoints"):
    """
    Visualize 3D keypoints using matplotlib
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='b', marker='o')
    
    # Add lines connecting joints for better visualization
    # You can customize these connections based on your keypoint order
    connections = [
        (0, 1), (1, 2), (2, 3),  # Head to spine
        (1, 4), (4, 5), (5, 6),  # Left arm
        (1, 7), (7, 8), (8, 9),  # Right arm
        (0, 10), (10, 11), (11, 12),  # Left leg
        (0, 13), (13, 14), (14, 15)   # Right leg
    ]
    
    for start, end in connections:
        xs = [keypoints[start, 0], keypoints[end, 0]]
        ys = [keypoints[start, 1], keypoints[end, 1]]
        zs = [keypoints[start, 2], keypoints[end, 2]]
        ax.plot(xs, ys, zs, 'r-')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more viewable
    ax.view_init(elev=20, azim=45)
    plt.show()

def visualize_smpl_mesh(vertices, faces, title="SMPL Mesh"):
    """
    Visualize SMPL mesh using pyrender
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Create pyrender scene
    scene = pyrender.Scene()
    
    # Add mesh to scene
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)
    
    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    np.sqrt(2)/2
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    
    # Render scene
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)
    
    # Display result
    plt.figure(figsize=(10, 10))
    plt.imshow(color)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compare_keypoints_and_joints(input_keypoints, smpl_joints, title="Keypoints vs SMPL Joints"):
    """
    Visualize both input keypoints and SMPL joints in the same plot
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot input keypoints
    ax.scatter(input_keypoints[:, 0], input_keypoints[:, 1], input_keypoints[:, 2], 
              c='b', marker='o', label='Input Keypoints')
    
    # Plot SMPL joints
    ax.scatter(smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2], 
              c='r', marker='x', label='SMPL Joints')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Make the plot more viewable
    ax.view_init(elev=20, azim=45)
    plt.show()

def reconstruct_3d_model(keypoints, height_m, weight_kg, visualize=True):
    """
    Main function to reconstruct 3D model from measurements with visualization
    """
    
    # Process keypoints
    processed_keypoints = process_keypoints(keypoints)
    
    if visualize:
        visualize_keypoints_3d(processed_keypoints, "Input MediaPipe Keypoints")
    
    # Create SMPL model
    model = create_smpl_model(gender='male')
    
    # Optimize SMPL parameters
    optimal_params = optimize_smpl_parameters(
        model,
        processed_keypoints,
        height_m,
        weight_kg
    )
    
    # Get final mesh
    pose = optimal_params[:72]
    betas = optimal_params[72:]
    
    output = model(
        betas=torch.tensor(betas).unsqueeze(0).float(),
        pose=torch.tensor(pose).unsqueeze(0).float()
    )
    
    vertices = output.vertices.detach().numpy()[0]
    faces = model.faces
    joints = output.joints.detach().numpy()[0]
    
    if visualize:
        visualize_smpl_mesh(vertices, faces, "Reconstructed SMPL Mesh")
        compare_keypoints_and_joints(processed_keypoints, joints, 
                                   "Comparison: Input Keypoints vs SMPL Joints")
    
    return {
        'vertices': vertices,
        'faces': faces,
        'joints': joints,
        'pose': pose,
        'betas': betas
    }

# Example usage
keypoints = np.array([
    [ 0.3145603, 0.44633162, -0.21216997],
    [ 0.30209484, 0.44210665, -0.21352102],
    [ 0.30185081, 0.43917693, -0.21385527],
    [ 0.30159964, 0.43634762, -0.2138051 ],
    [ 0.30269018, 0.45025301, -0.21486521],
    [ 0.30294129, 0.45265235, -0.21526813],
    [ 0.30310927, 0.45455609, -0.21541387],
    [ 0.30770116, 0.43665059, -0.17921506],
    [ 0.30978302, 0.45841311, -0.18554985],
    [ 0.32835931, 0.44099591, -0.19186139],
    [ 0.32869799, 0.45028332, -0.19581195],
    [ 0.39737961, 0.42231498, -0.13178604],
    [ 0.39973563, 0.47556847, -0.14087931],
    [ 0.48723451, 0.40051529, -0.10182737],
    [ 0.49229354, 0.50282921, -0.15451899],
    [ 0.55618756, 0.36624939, -0.14381481],
    [ 0.56361172, 0.53568373, -0.23777551],
    [ 0.57077455, 0.35376681, -0.16087891],
    [ 0.58185818, 0.54976035, -0.27038378],
    [ 0.56957005, 0.35190924, -0.19328019],
    [ 0.58008097, 0.5498295, -0.29760544],
    [ 0.56683495, 0.35784312, -0.15663328],
    [ 0.57504387, 0.54354248, -0.25277351],
    [ 0.60102994, 0.4357325, 0.0031446 ],
    [ 0.60033079, 0.46810544, -0.00339546],
    [ 0.74727007, 0.43245418, 0.09080669],
    [ 0.74791374, 0.47873236, 0.0389504 ],
    [ 0.88023049, 0.4277397, 0.25102477],
    [ 0.88216962, 0.48410095, 0.19920404],
    [ 0.89973473, 0.43004684, 0.25855622],
    [ 0.89953193, 0.48201056, 0.20986406],
    [ 0.91007085, 0.42777516, 0.15408912],
    [ 0.9113209, 0.49075196, 0.10963565]
]
)

result = reconstruct_3d_model(
    keypoints=keypoints,
    height_m=1.6764,
    weight_kg=58,
    visualize=false
)

# create trimesh object
mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces)

# visualize the mesh
mesh.show()
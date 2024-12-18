import numpy as np

def ray_box_collision(ray_origin, ray_directions, box_centers, box_extents):
    """
    Detect ray collisions with bounding boxes.

    Args:
        ray_origin (np.ndarray): Origin of the ray (shape: (3,)).
        ray_directions (np.ndarray): Unit length ray directions (shape: (N, 3)).
        box_centers (np.ndarray): Bounding box centers (shape: (M, 3)).
        box_extents (np.ndarray): Bounding box extents (half-widths, shape: (M, 3)).

    Returns:
        list of tuple: A list of hits in the format (box_id, face_id), where
                       box_id is the index of the box and face_id is the index of the face hit.
    """
    hits = []
    
    # Labeling scheme for faces: 0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z

    for box_id, (center, extent) in enumerate(zip(box_centers, box_extents)):
        # Compute the min and max bounds of the box
        box_min = center - extent
        box_max = center + extent

        # Ray-box intersection: Slabs method
        t_min = (box_min - ray_origin) / ray_directions
        t_max = (box_max - ray_origin) / ray_directions

        # Reorder t_min and t_max such that t_min <= t_max
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        # Find the largest t_min and smallest t_max for the slab intersection
        t_enter = np.max(t1, axis=1)
        t_exit = np.min(t2, axis=1)

        # Check for intersection
        intersect_mask = t_enter <= t_exit

        for ray_id, does_intersect in enumerate(intersect_mask):
            if does_intersect:
                # Find the face index that was hit
                t_min_indices = np.where(t1[ray_id] == t_enter[ray_id])[0]
                face_id = t_min_indices[0] * 2 if t_enter[ray_id] > 0 else t_min_indices[0] * 2 + 1
                hits.append((box_id, face_id))

    return hits

def compute_face_confidence(box, poses, detection_masks, detection_scores):
    """
    Compute confidence scores for each face of a bounding box.

    Args:
        box (tuple): Bounding box as (center, extents), where:
            - center (np.ndarray): Center of the box (shape: (3,)).
            - extents (np.ndarray): Half-dimensions of the box (shape: (3,)).
        poses (list of np.ndarray): List of poses for detections (shape: (4, 4) for each pose).
        detection_masks (list of np.ndarray): List of detection masks (binary masks, shape matches the image).
        detection_scores (list of float): List of confidence scores for each detection.

    Returns:
        np.ndarray: Confidence scores for each face of the bounding box (shape: (6,)).
    """
    center, extents = box
    face_confidences = np.zeros(6)

    # Loop through each detection
    for pose, mask, score in zip(poses, detection_masks, detection_scores):
        # Transform bounding box corners to camera space
        box_min = center - extents
        box_max = center + extents
        corners = np.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
        ])
        corners = np.dot(pose[:3, :3], corners.T).T + pose[:3, 3]

        # Define face vertices
        faces = [
            [corners[0], corners[2], corners[4], corners[6]],  # -X face
            [corners[1], corners[3], corners[5], corners[7]],  # +X face
            [corners[0], corners[1], corners[4], corners[5]],  # -Y face
            [corners[2], corners[3], corners[6], corners[7]],  # +Y face
            [corners[0], corners[1], corners[2], corners[3]],  # -Z face
            [corners[4], corners[5], corners[6], corners[7]],  # +Z face
        ]

        face_areas = np.zeros(6)
        for i, face in enumerate(faces):
            # Project face vertices to 2D (image plane)
            face_proj = np.dot(pose[:3, :3], np.array(face).T).T + pose[:3, 3]
            face_proj_2d = face_proj[:, :2] / face_proj[:, 2:3]  # Perspective divide

            # Compute area of the face in the image
            edge1 = face_proj_2d[1] - face_proj_2d[0]
            edge2 = face_proj_2d[3] - face_proj_2d[0]
            face_areas[i] = 0.5 * np.abs(np.cross(edge1, edge2))

        # Normalize face areas by the total mask area in the image
        face_area_ratios = face_areas / np.sum(face_areas)

        # Compute mask area ratio
        mask_area_ratio = np.sum(mask) / mask.size

        # Weight faces by detection score, mask area ratio, and actual area ratio
        face_confidences += face_area_ratios * score * mask_area_ratio

    return face_confidences

def compute_face_features(box, poses, detection_masks, detection_features):
    """
    Compute average features for each face of a bounding box.

    Args:
        box (tuple): Bounding box as (center, extents), where:
            - center (np.ndarray): Center of the box (shape: (3,)).
            - extents (np.ndarray): Half-dimensions of the box (shape: (3,)).
        poses (list of np.ndarray): List of poses for detections (shape: (4, 4) for each pose).
        detection_masks (list of np.ndarray): List of detection masks (binary masks, shape matches the image).
        detection_features (list of np.ndarray): List of feature vectors for each detection (shape: (F,)).

    Returns:
        dict: A dictionary mapping each face index (0-5) to the averaged feature vector (shape: (F,)).
    """
    center, extents = box
    face_features = {i: [] for i in range(6)}

    # Loop through each detection
    for pose, mask, features in zip(poses, detection_masks, detection_features):
        # Transform bounding box corners to camera space
        box_min = center - extents
        box_max = center + extents
        corners = np.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
        ])
        corners = np.dot(pose[:3, :3], corners.T).T + pose[:3, 3]

        # Define face vertices
        faces = [
            [corners[0], corners[2], corners[4], corners[6]],  # -X face
            [corners[1], corners[3], corners[5], corners[7]],  # +X face
            [corners[0], corners[1], corners[4], corners[5]],  # -Y face
            [corners[2], corners[3], corners[6], corners[7]],  # +Y face
            [corners[0], corners[1], corners[2], corners[3]],  # -Z face
            [corners[4], corners[5], corners[6], corners[7]],  # +Z face
        ]

        face_areas = np.zeros(6)
        for i, face in enumerate(faces):
            # Project face vertices to 2D (image plane)
            face_proj = np.dot(pose[:3, :3], np.array(face).T).T + pose[:3, 3]
            face_proj_2d = face_proj[:, :2] / face_proj[:, 2:3]  # Perspective divide

            # Compute area of the face in the image
            edge1 = face_proj_2d[1] - face_proj_2d[0]
            edge2 = face_proj_2d[3] - face_proj_2d[0]
            face_areas[i] = 0.5 * np.abs(np.cross(edge1, edge2))

        # Normalize face areas by the total mask area in the image
        face_area_ratios = face_areas / np.sum(face_areas)

        # Assign features to each face based on visibility ratio
        for i, ratio in enumerate(face_area_ratios):
            if ratio > 0:  # Only consider faces with non-zero visibility
                face_features[i].append(features * ratio)

    # Average features for each face
    averaged_features = {i: np.mean(face_features[i], axis=0) if face_features[i] else None for i in range(6)}

    return averaged_features

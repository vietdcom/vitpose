import torch
import numpy as np
import cv2
from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

import supervision as sv

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    # Load RT-DETR
    detector_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    detector_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device).eval()

    # Load ViTPose
    pose_processor = AutoProcessor.from_pretrained("yonigozlan/synthpose-vitpose-huge-hf")
    pose_model = VitPoseForPoseEstimation.from_pretrained("yonigozlan/synthpose-vitpose-huge-hf").to(device).eval()

    return detector_processor, detector_model, pose_processor, pose_model

def expand_box(box, image_width, image_height, scale=1.2):
    x1, y1, w, h = box
    cx, cy = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * scale, h * scale
    new_x1 = max(0, cx - new_w / 2)
    new_y1 = max(0, cy - new_h / 2)
    new_w = min(new_w, image_width - new_x1)
    new_h = min(new_h, image_height - new_y1)
    return [new_x1, new_y1, new_w, new_h]

def detect_humans(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=0.5
    )
    result = results[0]
    boxes = result["boxes"][result["labels"] == 0]

    if boxes.shape[0] == 0:
        return None

    boxes = boxes.cpu().numpy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    # Expand bounding boxes for better pose detection
    boxes = np.array([expand_box(box, image.width, image.height) for box in boxes])

    return boxes

def estimate_pose(image, boxes, processor, model):
    inputs = processor(image, boxes=[boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = processor.post_process_pose_estimation(outputs, boxes=[boxes])
    return pose_results[0]

def draw_pose_and_skeleton(image, pose_result):
    image = np.array(image.copy())

    for person in pose_result:
        keypoints = person["keypoints"].cpu().numpy()
        scores = person["scores"].cpu().numpy()

        # Vẽ keypoints
        for (x, y), conf in zip(keypoints, scores):
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Skeleton connections (COCO format 17 keypoints)
        skeleton = [
            (0, 1), (0, 2),  # Nose to Eyes
            (1, 3), (2, 4),  # Eyes to Ears
            (5, 6), # Shoulders
            (5, 7), (7, 9),  # Left Arm
            (6, 8), (8, 10),  # Right Arm
            (5, 11), (6, 12),  # Shoulders to Hips
            (11, 13), (13, 15),  # Left Leg
            (12, 14), (14, 16),  # Right Leg
            (11, 12)  # Hip center
        ]

        for idx1, idx2 in skeleton:
            if scores[idx1] > 0.5 and scores[idx2] > 0.5:
                pt1 = tuple(map(int, keypoints[idx1]))
                pt2 = tuple(map(int, keypoints[idx2]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)

    return Image.fromarray(image)

def main():
    detector_processor, detector_model, pose_processor, pose_model = load_models()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("✅ Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes = detect_humans(pil_image, detector_processor, detector_model)
        if boxes is None:
            cv2.imshow("ViTPose - No person detected", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        pose_result = estimate_pose(pil_image, boxes, pose_processor, pose_model)
        annotated = draw_pose_and_skeleton(pil_image, pose_result)

        result_frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        cv2.imshow("RT-DETR + ViTPose", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

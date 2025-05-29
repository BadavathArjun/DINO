import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DinoImageProcessor, DinoForObjectDetection

def visualize_predictions_cv2(image_path, model_path, threshold=0.5):
    """
    Visualize predictions using OpenCV
    """
    # Load the model and processor
    model = DinoForObjectDetection.from_pretrained(model_path)
    processor = DinoImageProcessor.from_pretrained(model_path)
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for processing
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process predictions
    target_sizes = torch.tensor([image.shape[:2]])
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]
    
    # Draw predictions on image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        score = score.item()
        
        if score > threshold:
            # Draw bounding box
            cv2.rectangle(
                image,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                2
            )
            
            # Prepare label text
            label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (box[0], box[1] - text_height - 10),
                (box[0] + text_width, box[1]),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label_text,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
    
    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Display image
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def real_time_detection(model_path, camera_id=0, threshold=0.5):
    """
    Perform real-time object detection using webcam
    """
    # Load the model and processor
    model = DinoForObjectDetection.from_pretrained(model_path)
    processor = DinoImageProcessor.from_pretrained(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(rgb_frame)
        inputs = processor(images=pil_image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process predictions
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        
        # Draw predictions on frame
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            score = score.item()
            
            if score > threshold:
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2
                )
                
                # Prepare label text
                label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (box[0], box[1] - text_height - 10),
                    (box[0] + text_width, box[1]),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label_text,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
        
        # Display FPS
        cv2.putText(
            frame,
            f"Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow("Real-time Object Detection", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def save_detection_video(input_video_path, output_video_path, model_path, threshold=0.5):
    """
    Process a video file and save the detection results
    """
    # Load the model and processor
    model = DinoForObjectDetection.from_pretrained(model_path)
    processor = DinoImageProcessor.from_pretrained(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(rgb_frame)
        inputs = processor(images=pil_image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process predictions
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]
        
        # Draw predictions on frame
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            score = score.item()
            
            if score > threshold:
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2
                )
                
                # Prepare label text
                label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (box[0], box[1] - text_height - 10),
                    (box[0] + text_width, box[1]),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label_text,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
        
        # Write frame to output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# For single image detection
visualize_predictions_cv2(
    image_path="path/to/your/image.jpg",
    model_path="./dino_object_detection",
    threshold=0.5
)

# For real-time webcam detection
real_time_detection(
    model_path="./dino_object_detection",
    camera_id=0,  # Use 0 for default webcam
    threshold=0.5
)

# For video processing
save_detection_video(
    input_video_path="input.mp4",
    output_video_path="output.mp4",
    model_path="./dino_object_detection",
    threshold=0.5
) 
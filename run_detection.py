import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

def run_webcam_detection():
    """
    Run real-time object detection using webcam
    """
    print("Initializing DETR model...")
    
    # Load the model and processor
    try:
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Starting webcam...")
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started successfully!")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(rgb_frame)
        
        # Process image
        inputs = processor(images=pil_image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process predictions
        target_sizes = torch.tensor([frame.shape[:2]])
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.5
        )[0]
        
        # Draw predictions on frame
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            score = score.item()
            
            if score > 0.5:  # Only show high confidence predictions
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
        
        # Add instructions text
        cv2.putText(
            frame,
            "Press 'q' to quit",
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
    print("Detection stopped")

if __name__ == "__main__":
    print("Starting Object Detection System...")
    run_webcam_detection() 
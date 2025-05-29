import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DinoForObjectDetection, DinoImageProcessor
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

def prepare_dataset():
    # Load COCO dataset
    dataset = load_dataset("detection-datasets/coco")
    
    # Initialize the image processor
    processor = DinoImageProcessor.from_pretrained("facebook/dino-base")
    
    def transform_images(example):
        # Process images and annotations
        image = example['image']
        annotations = example['objects']
        
        # Convert annotations to the format expected by DINO
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
        
        # Process image and annotations
        inputs = processor(
            images=image,
            annotations={
                "boxes": boxes,
                "labels": labels
            },
            return_tensors="pt"
        )
        
        return {
            "pixel_values": inputs["pixel_values"][0],
            "pixel_mask": inputs["pixel_mask"][0],
            "labels": inputs["labels"][0]
        }
    
    # Apply transformations
    dataset = dataset.map(transform_images, batched=False)
    return dataset

def train_model(model, train_loader, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

def evaluate_model(model, eval_loader, processor):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask
            )
            
            # Process predictions
            target_sizes = torch.tensor([pixel_values.shape[-2:]])
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )
            
            all_predictions.extend(results)
    
    return all_predictions

def main():
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    dataset = prepare_dataset()
    
    # Create data loaders
    train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
    eval_loader = DataLoader(dataset["validation"], batch_size=8)
    
    # Initialize model and processor
    print("Initializing DINO model...")
    model = DinoForObjectDetection.from_pretrained("facebook/dino-base")
    processor = DinoImageProcessor.from_pretrained("facebook/dino-base")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, optimizer)
    
    # Evaluate the model
    print("Evaluating model...")
    predictions = evaluate_model(model, eval_loader, processor)
    
    # Save the model
    print("Saving model...")
    model.save_pretrained("./dino_object_detection")
    processor.save_pretrained("./dino_object_detection")

if __name__ == "__main__":
    main() 
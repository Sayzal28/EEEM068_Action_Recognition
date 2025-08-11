import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset_splits(dataset_root_path, random_state=42):
    """
    Split dataset: 10% test, 72% train, 18% val
    """
    
    # Collect all videos
    all_videos = []
    action_classes = [d for d in os.listdir(dataset_root_path) 
                     if os.path.isdir(os.path.join(dataset_root_path, d))]
    
    for action_class in action_classes:
        class_path = os.path.join(dataset_root_path, action_class)
        videos = [v for v in os.listdir(class_path) 
                 if os.path.isdir(os.path.join(class_path, v))]
        
        for video in videos:
            all_videos.append({'class': action_class, 'video_name': video})
    
    df = pd.DataFrame(all_videos)
    
    # Split: 90% (train+val) and 10% test
    train_val, test = train_test_split(
        df, test_size=0.1, stratify=df['class'], random_state=random_state
    )
    
    # Split: 80% train, 20% val from remaining 90%
    train, val = train_test_split(
        train_val, test_size=0.2, stratify=train_val['class'], random_state=random_state
    )
    
    # Save CSV files
    train.to_csv('train.csv', index=False)
    val.to_csv('val.csv', index=False)
    test.to_csv('test.csv', index=False)
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

# Usage
dataset_path = "/path/to/your/dataset"  # Update this path
train_df, val_df, test_df = create_dataset_splits(dataset_path)

import sys
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from safetensors.torch import load_file
import glob
import pandas as pd
from tqdm import tqdm

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
sys.path.append(SRC_DIR)

from models import DeepfakeDetector
from config import Config
import video_inference

def get_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_test_videos.py <dataset_path>")
        return

    dataset_path = sys.argv[1]
    
    # Custom model support
    model_name = "patched_model.safetensors"
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
        
    if not os.path.exists(dataset_path):
        print(f"Error: Path not found: {dataset_path}")
        return

    print(f"Scanning dataset at: {dataset_path}")
    
    # Find all videos
    extensions = ('*.mp4', '*.avi', '*.mov', '*.webm', '*.mkv')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} videos.")

    # Load Model
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    print(f"Loading Model: {model_name}")
    
    model = DeepfakeDetector(pretrained=True) 
    
    # Check if absolute path or just filename
    if os.path.isabs(model_name) or "/" in model_name:
        checkpoint_path = model_name
    else:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, model_name)
    
    if os.path.exists(checkpoint_path):
         state_dict = load_file(checkpoint_path)
         model.load_state_dict(state_dict, strict=False)
         print("✅ Model loaded.")
    else:
         print(f"⚠ Model not found at {checkpoint_path}")
         return

    model.to(device)
    model.eval()
    transform = get_transform()

    results = []

    print("-" * 90)
    print(f"{'Filename':<30} | {'GT':<5} | {'Verdict':<8} | {'Conf':<6} | {'Avg':<6} | {'Max':<6} | {'SuspFrames'}")
    print("-" * 90)

    try:
        for video_path in tqdm(video_files, desc="Processing"):
            try:
                # Run inference (fast mode: 1 fps)
                res = video_inference.process_video(video_path, model, transform, device, frames_per_second=1)
                
                if "error" in res:
                    print(f"Error processing {os.path.basename(video_path)}: {res['error']}")
                    continue
                    
                filename = os.path.basename(video_path)
                
                # Infer Ground Truth
                lower_path = video_path.lower()
                if "real" in lower_path:
                    ground_truth = "REAL"
                elif "fake" in lower_path:
                    ground_truth = "FAKE"
                else:
                    ground_truth = "UNKNOWN"

                suspicious = len(res.get('suspicious_frames', []))
                
                results.append({
                    "Filename": filename,
                    "GroundTruth": ground_truth,
                    "Verdict": res['prediction'],
                    "Confidence": res['confidence'],
                    "AvgProb": res['avg_fake_prob'],
                    "MaxProb": res['max_fake_prob'],
                    "SuspiciousFrames": suspicious
                })
                
                # Print row
                gt_str = ground_truth if ground_truth != "UNKNOWN" else "?"
                print(f"{filename[:30]:<30} | {gt_str:<5} | {res['prediction']:<8} | {res['confidence']:.2f}   | {res['avg_fake_prob']:.2f}   | {res['max_fake_prob']:.2f}   | {suspicious}")
                
            except Exception as e:
                print(f"Failed {video_path}: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Generating summary for processed videos so far...")

    # Summary
    print("\n" + "=" * 30)
    print(" SUMMARY")
    print("=" * 30)
    df = pd.DataFrame(results)
    if not df.empty:
        total = len(df)
        fakes = len(df[df['Verdict'] == 'FAKE'])
        reals = len(df[df['Verdict'] == 'REAL'])
        print(f"Total Videos: {total}")
        print(f"Detected FAKE: {fakes} ({(fakes/total)*100:.1f}%)")
    if not df.empty:
        total = len(df)
        fakes = len(df[df['Verdict'] == 'FAKE'])
        reals = len(df[df['Verdict'] == 'REAL'])
        
        print(f"Total Videos: {total}")
        print(f"Detected FAKE: {fakes} ({(fakes/total)*100:.1f}%)")
        print(f"Detected REAL: {reals} ({(reals/total)*100:.1f}%)")
        
        # Calculate Metrics
        valid_df = df[df['GroundTruth'] != 'UNKNOWN']
        if not valid_df.empty:
            tp = len(valid_df[(valid_df['GroundTruth'] == 'FAKE') & (valid_df['Verdict'] == 'FAKE')])
            tn = len(valid_df[(valid_df['GroundTruth'] == 'REAL') & (valid_df['Verdict'] == 'REAL')])
            fp = len(valid_df[(valid_df['GroundTruth'] == 'REAL') & (valid_df['Verdict'] == 'FAKE')])
            fn = len(valid_df[(valid_df['GroundTruth'] == 'FAKE') & (valid_df['Verdict'] == 'REAL')])
            
            accuracy = (tp + tn) / len(valid_df) * 100
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("-" * 30)
            print(" METRICS (on labeled data)")
            print("-" * 30)
            print(f"Accuracy:  {accuracy:.2f}%")
            print(f"Precision: {precision:.2f}%")
            print(f"Recall:    {recall:.2f}%")
            print(f"F1 Score:  {f1:.2f}%")
            print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Save CSV
        output_csv = "video_batch_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    main()

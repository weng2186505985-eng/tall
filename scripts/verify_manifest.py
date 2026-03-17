import json
from pathlib import Path

def verify_manifest(manifest_path):
    print(f"Checking manifest: {manifest_path}")
    if not Path(manifest_path).exists():
        print("Error: Manifest file not found!")
        return

    with open(manifest_path, 'r') as f:
        data = json.load(f)

    all_data = data.get('all_preprocessed', [])
    print(f"Total videos in manifest: {len(all_data)}")
    
    missing_files = 0
    class_counts = {}
    
    for item in all_data:
        fake_type = item.get('fake_type', 'Original') if item.get('label') == 1 else 'Original'
        class_counts[fake_type] = class_counts.get(fake_type, 0) + 1
        
        for frame_path in item.get('frame_paths', []):
            if not Path(frame_path).exists():
                missing_files += 1
    
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    if missing_files > 0:
        print(f"\nWarning: {missing_files} frame files are missing from disk!")
    else:
        print("\nSuccess: All referenced frame files exist on disk.")

if __name__ == "__main__":
    manifest_file = "deepfake_detection/data/processed/dataset_manifest.json"
    verify_manifest(manifest_file)

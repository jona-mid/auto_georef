import json
from pathlib import Path

def main():
    output_dir = Path("data/raw/dataset_custom")
    
    # find all unique IDs
    ids = set()
    for f in output_dir.glob("*_ortho_streets.png"):
        ortho_id = f.name.split('_')[0]
        ids.add(ortho_id)
        
    captured = [{"ortho_id": i, "url": f"https://deadtrees.earth/dataset/{i}"} for i in ids]
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(captured, f, indent=2)

    # Create empty labels.csv
    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("ortho_id,label\n")
        for item in captured:
            f.write(f"{item['ortho_id']},\n")
            
    print(f"Generated metadata and labels for {len(ids)} items")

if __name__ == "__main__":
    main()

"""
Test SuperPoint+LightGlue matching on scraped ortho images.
"""

import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def main():
    # Add parent to path to find georef_check module
    parent = Path(__file__).parent.parent
    sys.path.insert(0, str(parent))

    # Import matching module
    matching_path = parent / "georef_check" / "src" / "features" / "matching.py"
    spec = spec_from_file_location("matching", matching_path)
    matching = module_from_spec(spec)
    spec.loader.exec_module(matching)

    SuperPointLightGlueMatcher = matching.SuperPointLightGlueMatcher
    load_image = matching.load_image

    # Disable tqdm
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_dir = parent / "georef_check" / "data" / "raw" / "dataset_custom"

    # Get all ortho IDs
    ortho_ids = sorted(
        set(
            f.stem.replace("_ortho_streets", "")
            .replace("_ortho_satellite", "")
            .replace("_streets_only", "")
            .replace("_satellite_only", "")
            for f in data_dir.glob("*_ortho_streets.png")
        )
    )

    if not ortho_ids:
        print("No ortho images found!")
        return

    print(f"Found {len(ortho_ids)} orthos: {ortho_ids[:5]}... (showing first 5)")
    print("\nTesting SuperPoint+LightGlue on all images...\n")

    # Initialize matcher
    print("Loading SuperPoint+LightGlue model...")
    matcher = SuperPointLightGlueMatcher(device="cpu")
    print("Model loaded!\n")

    results = []
    for ortho_id in ortho_ids:
        ortho_streets = data_dir / f"{ortho_id}_ortho_streets.png"
        streets_only = data_dir / f"{ortho_id}_streets_only.png"
        ortho_satellite = data_dir / f"{ortho_id}_ortho_satellite.png"
        satellite_only = data_dir / f"{ortho_id}_satellite_only.png"

        if not all(
            p.exists()
            for p in [ortho_streets, streets_only, ortho_satellite, satellite_only]
        ):
            print(f"Skipping {ortho_id} - missing images")
            continue

        # Match streets
        ortho_streets_img = load_image(str(ortho_streets))
        streets_only_img = load_image(str(streets_only))
        streets_result = matcher.match_images(ortho_streets_img, streets_only_img)

        # Match satellite
        ortho_satellite_img = load_image(str(ortho_satellite))
        satellite_only_img = load_image(str(satellite_only))
        satellite_result = matcher.match_images(ortho_satellite_img, satellite_only_img)

        combined_prob = (
            streets_result.good_probability + satellite_result.good_probability
        ) / 2.0

        results.append(
            {
                "id": ortho_id,
                "streets_prob": streets_result.good_probability,
                "satellite_prob": satellite_result.good_probability,
                "combined_prob": combined_prob,
                "streets_inliers": streets_result.num_inliers,
                "satellite_inliers": satellite_result.num_inliers,
            }
        )

        print(
            f"{ortho_id}: streets={streets_result.good_probability:.3f} ({streets_result.num_inliers} inliers), "
            f"satellite={satellite_result.good_probability:.3f} ({satellite_result.num_inliers} inliers), "
            f"combined={combined_prob:.3f}"
        )

    # Summary
    probs = [r["combined_prob"] for r in results]
    print(f"\n=== Summary ===")
    print(f"Tested {len(results)} orthos")
    print(f"Mean good_probability: {sum(probs) / len(probs):.3f}")
    print(f"Min: {min(probs):.3f}, Max: {max(probs):.3f}")


if __name__ == "__main__":
    main()

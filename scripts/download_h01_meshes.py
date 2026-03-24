import argparse
from pathlib import Path
import trimesh
import cloudvolume


def download_mesh(seg_id: int, out_dir: Path, cv):
    print(f"\nDownloading mesh for segment {seg_id}...")

    mesh_data = cv.mesh.get(seg_id)

    if not mesh_data:
        print(f"No mesh found for {seg_id}")
        return False

    first_mesh = list(mesh_data.values())[0]
    vertices = first_mesh.vertices
    faces = first_mesh.faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_path = out_dir / f"h01_mesh_{seg_id}.ply"
    mesh.export(out_path)

    print(f"Saved: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download H01 neuron meshes")
    parser.add_argument(
        "--segment-ids",
        type=int,
        nargs="+",
        required=True,
        help="List of H01 segment IDs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\91813\Documents\github\mitsuba-neuron-tinkering\neuron",
        help="Where to save meshes",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to H01 dataset...")
    cv = cloudvolume.CloudVolume(
        "gs://h01-release/data/20210601/c3",
        progress=True
    )

    for seg_id in args.segment_ids:
        try:
            download_mesh(seg_id, out_dir, cv)
        except Exception as e:
            print(f"Error for {seg_id}: {e}")


if __name__ == "__main__":
    main()
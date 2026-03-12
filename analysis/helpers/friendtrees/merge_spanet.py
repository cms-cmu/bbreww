import contextlib
import fnmatch
import glob
import os
import subprocess
import tempfile

import h5py
import numpy as np

_EOS_XRD = "root://cmseos.fnal.gov/"
# FUSE mount -> xrootd prefix mapping
_FUSE_TO_XRD = {
    "/eos/uscms/store/": f"{_EOS_XRD}/store/",
}


def _to_xrd(path: str) -> str:
    """Convert an EOS FUSE mount path to its xrootd equivalent."""
    for fuse, xrd in _FUSE_TO_XRD.items():
        if path.startswith(fuse):
            return path.replace(fuse, xrd, 1)
    return path


def _list_xrd(xrd_dir: str, pattern: str) -> list[str]:
    """List files on EOS via xrdfs that match a glob pattern."""
    # extract the server and the directory path
    # xrd_dir looks like root://cmseos.fnal.gov//store/user/...
    server = xrd_dir.split("//")[0] + "//" + xrd_dir.split("//")[1]
    dir_path = "/" + xrd_dir.split("//", 2)[2]
    dir_path = dir_path.rstrip("/")

    result = subprocess.run(
        ["xrdfs", server, "ls", dir_path],
        capture_output=True, text=True, check=True,
    )
    all_files = sorted(result.stdout.strip().splitlines())
    matched = [
        f"{server}/{f}"
        for f in all_files
        if fnmatch.fnmatch(os.path.basename(f), pattern)
    ]
    return matched


def _is_eos(path: str) -> bool:
    """Check if a path points to EOS (either FUSE mount or xrootd)."""
    return path.startswith("root://") or any(
        path.startswith(fuse) for fuse in _FUSE_TO_XRD
    )


@contextlib.contextmanager
def _open_h5(path: str):
    """Open an HDF5 file, copying from xrootd to a temp file if needed."""
    if path.startswith("root://"):
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            subprocess.run(
                ["xrdcp", "-f", path, tmp.name],
                check=True, capture_output=True, text=True,
            )
            with h5py.File(tmp.name, "r") as f:
                yield f
    else:
        with h5py.File(path, "r") as f:
            yield f


def merge_spanet_h5(input_dir: str, output_file: str):
    """
    Merge per-chunk SPANet HDF5 files into a single file.

    Supports both local paths and EOS paths (xrootd or FUSE mount).

    Parameters
    ----------
    input_dir : str
        Directory containing ``spanet_*.h5`` chunk files.
        Can be a local path, FUSE mount, or xrootd path
        (e.g. ``root://cmseos.fnal.gov//store/user/...``).
    output_file : str
        Path for the merged output file (local).
    """
    if _is_eos(input_dir):
        xrd_dir = _to_xrd(input_dir)
        chunk_files = _list_xrd(xrd_dir, "spanet_*.h5")
    else:
        chunk_files = sorted(glob.glob(os.path.join(input_dir, "spanet_*.h5")))

    if not chunk_files:
        raise FileNotFoundError(f"No spanet_*.h5 files found in {input_dir}")

    # find the common dataset keys across all chunk files
    common_keys = None
    for path in chunk_files:
        with _open_h5(path) as f:
            keys = set(_get_structure(f))
            common_keys = keys if common_keys is None else common_keys & keys
    structure = sorted(common_keys)

    # collect all data, determining expected shape from the first file
    data = {key: [] for key in structure}
    expected_shape = {}  # trailing shape after the event axis
    for i, path in enumerate(chunk_files):
        with _open_h5(path) as f:
            for key in structure:
                arr = f[key][:]
                if i == 0:
                    expected_shape[key] = arr.shape[1:]
                elif arr.shape[1:] != expected_shape[key]:
                    # reshape to match: squeeze extra dims or expand missing ones
                    target = (arr.shape[0], *expected_shape[key])
                    arr = arr.reshape(target)
                data[key].append(arr)

    # concatenate
    for key in structure:
        data[key] = np.concatenate(data[key], axis=0)

    n_before = data[structure[0]].shape[0]

    # deduplicate using a composite key of event kinematics
    dedup_keys = [
        "INPUTS/lepton/pt", "INPUTS/lepton/eta", "INPUTS/lepton/phi",
        "INPUTS/met/px", "INPUTS/met/py", "INPUTS/event/HT",
    ]
    dedup_keys = [k for k in dedup_keys if k in data]
    if dedup_keys:
        composite = np.column_stack([data[k] for k in dedup_keys])
        dtype = np.dtype((np.void, composite.dtype.itemsize * composite.shape[1]))
        _, unique_idx = np.unique(
            np.ascontiguousarray(composite).view(dtype).ravel(),
            return_index=True,
        )
        unique_idx = np.sort(unique_idx)
        if len(unique_idx) < n_before:
            for key in structure:
                data[key] = data[key][unique_idx]
            print(f"Removed {n_before - len(unique_idx)} duplicate events")

    # write
    with h5py.File(output_file, "w") as f:
        for key in structure:
            f.create_dataset(key, data=data[key])

    n_events = data[structure[0]].shape[0]
    print(f"Merged {len(chunk_files)} files -> {output_file} ({n_events} events)")


def _get_structure(f: h5py.File, prefix: str = "") -> list[str]:
    """Recursively get all dataset paths in an HDF5 file."""
    paths = []
    for key in f.keys():
        full = f"{prefix}/{key}" if prefix else key
        if isinstance(f[key], h5py.Dataset):
            paths.append(full)
        else:
            paths.extend(_get_structure(f[key], full))
    return paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_dir> <output_file>")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} /eos/uscms/store/user/akhanal/HHbbWW/2025_v5/spanet_output/ spanet_merged.h5")
        print(f"  python {sys.argv[0]} root://cmseos.fnal.gov//store/user/akhanal/HHbbWW/2025_v5/spanet_output/ spanet_merged.h5")
        sys.exit(1)

    merge_spanet_h5(sys.argv[1], sys.argv[2])

    with h5py.File(sys.argv[2], "r+") as f:
        # Signal = events where both higgs_bb and higgs_WW have valid assignments
        
        sig_mask = f["TARGETS/higgs_bb/MASK"][:].astype(bool) & f["TARGETS/higgs_WW/MASK"][:].astype(bool)
        
        w = f["EVENT_WEIGHT"][:]
        
        # Normalize each process so weights sum to 1
        sig_sum = np.abs(w[sig_mask]).sum()
        bkg_sum = np.abs(w[~sig_mask]).sum()
        
        w[sig_mask] /= sig_sum
        w[~sig_mask] /= bkg_sum
        
        f["EVENT_WEIGHT"][:] = w
        
        print(f"Signal events: {sig_mask.sum()}, weight sum before: {sig_sum:.2f}, after: {np.abs(w[sig_mask]).sum():.4f}")
        print(f"Background events: {(~sig_mask).sum()}, weight sum before: {bkg_sum:.2f}, after: {np.abs(w[~sig_mask]).sum():.4f}")


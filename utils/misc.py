import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union


def _to_builtin_python(obj):
    """
    Recursively convert NumPy types and arrays to native Python types for JSON serialization.

    Args:
        obj: Object to convert, which may be a NumPy scalar, array, or nested container.

    Returns:
        The equivalent Python-native object:
        - NumPy integers → int
        - NumPy floats → float
        - NumPy arrays → list (recursively converted)
        - dict → dict with string keys and converted values
        - list, tuple, set → list with converted elements
        - Other types are returned unchanged
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_builtin_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_builtin_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin_python(x) for x in obj]
    return obj


def get_file_paths_to_save(save_root: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Prepare file paths for saving scaling results.

    Creates a timestamped subfolder inside `save_root` and returns paths for
    both JSON and PDF files.

    Args:
        save_root (str | Path): Root directory where results should be saved.

    Returns:
        Tuple[Path, Path]: Tuple containing the JSON and PDF file paths:
            (file_path_json, file_path_pdf)
    """
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)  # ensure root exists

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = save_root / f"{timestamp}"
    subfolder.mkdir(parents=True, exist_ok=True)

    # Paths for JSON and PDF inside the subfolder
    file_path_json = subfolder / "results.json"
    file_path_pdf = subfolder / "results.pdf"

    return file_path_json, file_path_pdf


def save_benchmark_results_json(
    save_path: Union[str, Path],
    results: Dict[str, Dict[str, Any]],
    dataset_sizes: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Save benchmark results and dataset sizes to a JSON file.

    Args:
        save_path (str | Path): File path where the results should be saved.
        results (dict): Nested dictionary of results in the form
            `structure_name -> metric_name -> list/values`.
        dataset_sizes: Sequence or NumPy array of dataset sizes used for the benchmark.
        meta (dict, optional): Optional metadata to include (e.g., parameters, seed, timestamp).

    Returns:
        Path | None: Path to the saved JSON file if successful, otherwise None.
    """
    try:
        file_path = Path(save_path)

        payload = {
            "meta": _to_builtin_python(meta or {}),
            "dataset_sizes": _to_builtin_python(dataset_sizes),
            "results": _to_builtin_python(results),
            "saved_at": datetime.now().isoformat(),
        }

        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        print(f"Saved benchmark results to: {file_path}")
        return file_path

    except Exception as exc:
        print(f"Failed to save benchmark results: {exc}")
        return None

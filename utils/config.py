import yaml


def load_config(path: str = "config/main.yaml") -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        path (str, optional): Path to the YAML config file. Defaults to "config/main.yaml".

    Returns:
        dict: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

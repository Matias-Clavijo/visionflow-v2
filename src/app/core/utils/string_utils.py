import re
import os


def to_snake_case(s: str) -> str:
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    return s


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    project_root = os.path.abspath(project_root)
    
    resolved_path = os.path.join(project_root, path)
    return os.path.abspath(resolved_path)
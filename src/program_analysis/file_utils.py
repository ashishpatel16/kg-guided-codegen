import json
import os


def dump_to_json(file_name: str, data: dict, dump_dir="artifacts"):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    file_path = os.path.join(dump_dir, file_name)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

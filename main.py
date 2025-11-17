def main():
    print("Hello from kg-guided-codegen!")

from benchmarks.dataset_loader import load_humaneval
from program_analysis.file_utils import dump_to_json
if __name__ == "__main__":
    data = load_humaneval()
    dataset = []
    for i in data:
        dataset.append(i)
    dump_to_json("humaneval.json", dataset)
    print(data[0], type(data[0]))


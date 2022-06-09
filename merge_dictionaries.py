import argparse
import pickle
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_paths",
    type=str, nargs="*",
    help="The paths of the input pickled dictionaries.",
)
parser.add_argument(
    "--output_path",
    type=str,
    help="The path where to save the resulting dictionary as a pickle file."
)
args = parser.parse_args()


def get_mtime(path: Path):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = path.stat()

    return ctime


if __name__ == "__main__":
    input_paths_str = args.input_paths
    input_paths = list(map(lambda path_str: Path(path_str), input_paths_str))
    all_input_paths_ok = True
    for input_path in input_paths:
        if not input_path.exists():
            all_input_paths_ok = False
            print(f"Error: input path does not exist: {str(input_path)}.")
    if not all_input_paths_ok:
        raise AssertionError(f"Some of the input dictionary paths were not found.")
    # Sort the input paths based on their creation time such that newer
    # dictionaries will have priority.
    input_paths.sort(key=get_mtime)

    output_path_str = args.output_path
    if output_path_str is None:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path_str = f"./logs/merged_logs_{timestamp}.pkl"
    output_path = Path(output_path_str)
    if output_path.exists():
        raise AssertionError(f"Output path {output_path_str} already exists.")

    output_dict = dict()
    for input_path in input_paths:
        with open(input_path, mode="rb") as input_file:
            input_dict = pickle.load(input_file)
            output_dict.update(input_dict)

    with open(output_path_str, mode="wb") as output_file:
        pickle.dump(output_dict, output_file)

    print(f"Combined, in this order, the following dictionaries:")
    for input_path in input_paths:
        print(f"  -> {str(input_path)}")
    print(f"Into the output dictionary at {output_path_str}.")

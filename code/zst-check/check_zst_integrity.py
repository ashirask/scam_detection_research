import os
import subprocess
import argparse

def check_zst_files(directory):
    """Check all .zst files in a directory for corruption."""
    corrupted_files = []
    valid_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".zst"):
                file_path = os.path.join(root, file)
                print(f"Checking: {file_path}")

                try:
                    # Run zstd -t to test the file
                    subprocess.run(["zstd", "-t", file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    valid_files.append(file_path)
                except subprocess.CalledProcessError:
                    print(f"Corrupted: {file_path}")
                    corrupted_files.append(file_path)

    return valid_files, corrupted_files

def main():
    parser = argparse.ArgumentParser(description="Check integrity of .zst files in a directory.")
    parser.add_argument("--directory", required=True, help="Path to the directory containing .zst files")
    parser.add_argument("--output", default="zst_integrity_report.txt", help="File to save the integrity report")
    args = parser.parse_args()

    valid_files, corrupted_files = check_zst_files(args.directory)

    # Save results to a file
    with open(args.output, "w") as report:
        report.write("=== Valid Files ===\n")
        report.writelines(f"{file}\n" for file in valid_files)

        report.write("\n=== Corrupted Files ===\n")
        report.writelines(f"{file}\n" for file in corrupted_files)

    print(f"Integrity check complete. Report saved to {args.output}")

if __name__ == "__main__":
    main()
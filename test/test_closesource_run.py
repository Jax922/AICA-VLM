import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/aica_vlm")))

print("Adding src/aica_vlm to sys.path")

from cli import closedmodel_benchmark

config_file_path = "examples/adaptation/closesource_models/understanding_zero_shot/gpt4o-mini.yaml"

print("Loading configuration file:", config_file_path)

def test_config_loader():
    """
    Test the ConfigLoader by loading a configuration file and printing the results.
    """
    closedmodel_benchmark(config_file_path)

if __name__ == "__main__":
    test_config_loader()
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/aica_vlm")))

import adaptation as adt

config_file_path = "examples/adaptation/qwen2.5VL.yaml"

def test_config_loader():
    """
    Test the ConfigLoader by loading a configuration file and printing the results.
    """
    adt.run(config_file_path)

if __name__ == "__main__":
    test_config_loader()
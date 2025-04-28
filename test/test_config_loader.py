import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/aica_vlm")))

from adaptation import ConfigLoader

config_file_path = "/home/pci/yxr/AICA-VLM/examples/adaptation/qwen2.5VL.yaml"

def test_config_loader():
    """
    Test the ConfigLoader by loading a configuration file and printing the results.
    """
    try:
        # Initialize the ConfigLoader
        config_loader = ConfigLoader(config_file_path)
        
        # Load the configuration
        config_loader.load()
        
        # Print the loaded model information and tasks
        print("Configuration loaded successfully!")
        print("Model Information:", config_loader.model_info)
        print("Tasks:", config_loader.tasks)
    except Exception as e:
        # Print any errors encountered during the process
        print("Error while loading configuration:", e)

if __name__ == "__main__":
    test_config_loader()
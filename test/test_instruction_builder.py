import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/aica_vlm")))

from instructions import InstructionBuilder

def test_instruction_builder():
    ins_builder = InstructionBuilder(
        instruction_type="EU_observer_emotion",
        dataset_path=r"D:\dev\VLM-EQ\datasets\benchmark\EU_observer_emotion_emoset",
        emotion_model="8_expanded"
    )

    ins_builder.build()

if __name__ == '__main__':
    print("Testing InstructionBuilder...")
    test_instruction_builder()
    print("InstructionBuilder test completed.")
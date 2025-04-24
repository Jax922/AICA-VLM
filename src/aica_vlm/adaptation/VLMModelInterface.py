import torch
from abc import ABC, abstractmethod
from PIL import Image
import requests

class VLMModelInterface(ABC):
    """抽象基类，定义统一的 VLM 模型接口"""

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def preprocess_input(self, image_path, question):
        """预处理输入"""
        pass

    @abstractmethod
    def infer(self, inputs):
        """执行推理"""
        pass

    @abstractmethod
    def postprocess_output(self, outputs):
        """后处理输出"""
        pass

    def predict(self, image_path, question):
        """统一的推理接口"""
        inputs = self.preprocess_input(image_path, question)
        outputs = self.infer(inputs)
        return self.postprocess_output(outputs)

    def predict_with_prompt(self, prompt):
        """使用统一的 prompt 进行推理"""
        inputs = self.preprocess_prompt(prompt)
        outputs = self.infer(inputs)
        return self.postprocess_output(outputs)

    @abstractmethod
    def preprocess_prompt(self, prompt):
        """预处理统一的 prompt"""
        pass


class MiniCPMV26(VLMModelInterface):
    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-V-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

    def preprocess_input(self, image_path, question):
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question]}]
        return msgs

    def preprocess_prompt(self, prompt):
        """将统一的 prompt 转换为模型输入格式"""
        image_path = prompt["images"][0]
        question = prompt["messages"][0]["content"]
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, question]}]
        return msgs

    def infer(self, inputs):
        return self.model.chat(image=None, msgs=inputs, tokenizer=self.tokenizer)

    def postprocess_output(self, outputs):
        return outputs


class LlavaOneVision(VLMModelInterface):
    def load_model(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        self.model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def preprocess_prompt(self, prompt):
        """将统一的 prompt 转换为模型输入格式"""
        image_path = prompt["images"][0]
        question = prompt["messages"][0]["content"]
        raw_image = Image.open(image_path).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
        return inputs

    def infer(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

    def postprocess_output(self, outputs):
        return self.processor.decode(outputs[0][2:], skip_special_tokens=True)

class Qwen2VL(VLMModelInterface):
    """Qwen2-VL 模型实现"""

    def load_model(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def preprocess_input(self, image_path, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[Image.open(image_path)], padding=True, return_tensors="pt"
        ).to("cuda")
        return inputs

    def infer(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=128)

    def postprocess_output(self, outputs):
        return self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# 统一推理接口
def unified_inference(model_class, image_path, question):
    model = model_class()
    model.load_model()
    result = model.predict(image_path, question)
    return result

if __name__ == "__main__":
    # 示例调用
    image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"
    question = "What is in the image?"

    # 使用 MiniCPM-V-2_6
    print("MiniCPM-V-2_6 Result:")
    print(unified_inference(MiniCPMV26, image_path, question))

    # 使用 Llava-OneVision
    print("Llava-OneVision Result:")
    print(unified_inference(LlavaOneVision, image_path, question))

    # 使用 Qwen2-VL
    print("Qwen2-VL Result:")
    print(unified_inference(Qwen2VL, image_path, question))

    import os
    import sys

    # 动态添加项目根目录到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.instructions.builder import InstructionBuilder

    # 初始化 InstructionBuilder
    instruction_builder = InstructionBuilder(
        instruction_type="CES",  # 指定模板类型
        dataset_path="./datasets/ArtEmis",  # 数据集路径
        emotion_model="CES"  # 情感模型类型
    )

    # 构建指令
    instruction_builder.build()
    prompts = instruction_builder.get_instructions()

    # 使用模型进行推理
    model = MiniCPMV26()  # 替换为其他模型类以测试不同模型
    model.load_model()

    for prompt in prompts[:5]:  # 示例推理前 5 条指令
        result = model.predict_with_prompt(prompt)
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")


    from src.aica_vlm.instructions.builder import InstructionBuilder

    # 初始化参数
    instruction_type = "CES"  # 指令类型，例如 "CES" 或 "VA"
    dataset_path = "./datasets/ArtEmis"  # 数据集根目录路径
    emotion_model = "CES"  # 情感模型名称，例如 "CES" 或 "VA"

    # 初始化 InstructionBuilder
    instruction_builder = InstructionBuilder(
        instruction_type=instruction_type,
        dataset_path=dataset_path,
        emotion_model=emotion_model
    )

    # 构建指令
    instruction_builder.build()

    # 获取生成的指令
    instructions = instruction_builder.get_instructions()

    # 打印部分生成的指令
    for instruction in instructions[:5]:  # 示例打印前 5 条指令
        print(instruction)

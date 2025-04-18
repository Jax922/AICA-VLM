import os
import csv
import re

# 指定图片文件夹路径
image_folder = "FI_dataset_images"
# 指定输出CSV文件路径
output_csv = "FI_dataset_annotations.csv"

# 创建CSV文件
with open(output_csv, 'w', newline='') as csvfile:
    # 定义CSV表头
    fieldnames = ['img_name', 'img_folder', 'emotion_cat', 'emotion_v', 'emotion_a', 'dataset_source']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入表头
    writer.writeheader()
    
    # 遍历文件夹中的所有图片
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.jpg'):
            # 使用正则表达式从文件名中提取标签（第一个下划线前的内容）
            match = re.match(r"([^_]+)_", filename)
            
            if match:
                emotion_category = match.group(1)
                
                # 创建并写入一行数据
                writer.writerow({
                    'img_name': filename,
                    'img_folder': 'emotion_dataset_images',
                    'emotion_cat': emotion_category,
                    'emotion_v': '',  # 空值
                    'emotion_a': '',  # 空值
                    'dataset_source': 'FI_dataset'
                })

print(f"CSV文件已创建: {output_csv}")
print(f"CSV processing completed successfully.")
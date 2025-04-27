# TwitterLDL dataset
# https://github.com/sherleens/EmotionDistributionLearning/blob/57eb79073e7750132e464292ac890b0dc4e02db2/README.md#download-dataset-lmdb


import lmdb
import os
import csv
import io
from tqdm import tqdm
from PIL import Image
import numpy as np

from google.protobuf import descriptor_pb2
from google.protobuf.message import DecodeError
from google.protobuf import message_factory
from google.protobuf import descriptor_pool

pool = descriptor_pool.Default()
file_descriptor_proto = descriptor_pb2.FileDescriptorProto()
file_descriptor_proto.name = 'datum.proto'
file_descriptor_proto.package = 'caffe'

message_type = file_descriptor_proto.message_type.add()
message_type.name = 'Datum'

fields = [
    ('int32', 'channels', 1),
    ('int32', 'height', 2),
    ('int32', 'width', 3),
    ('bytes', 'data', 4),
    ('repeated float', 'float_data', 6),
    ('bool', 'encoded', 7)
]

for type_str, name, number in fields:
    field = message_type.field.add()
    field.name = name
    field.number = number
    field.label = 1 if not type_str.startswith('repeated') else 3
    field.type = {
        'int32': 5,
        'bytes': 12,
        'float': 2,
        'bool': 8
    }[type_str.replace('repeated ', '')]

file_desc = pool.Add(file_descriptor_proto)
Datum = message_factory.MessageFactory(pool).GetPrototype(file_desc.message_types_by_name['Datum'])

lmdb_path = "/home/pci/yxr/AICA-VLM/datasets/TwitterLDL/TwitterLDL/train_twitterldl_split1_lmdb"
output_image_dir = "output_images"
output_csv_path = "twitterldl_final.csv"
dataset_source = "TwitterLDL"
img_folder = "images" 

os.makedirs(output_image_dir, exist_ok=True)

# === Mikel's Emotion ===
emotion_names = [
    "Amusement",
    "Contentment",
    "Awe",
    "Excitement",
    "Fear",
    "Sadness",
    "Disgust",
    "Anger"
]

# === LMDB ===
env = lmdb.open(lmdb_path, readonly=True, lock=False)
txn = env.begin()
cursor = txn.cursor()

with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["img_name", "img_folder", "emotion_cat", "emotion_v", "emotion_a", "source_dataset"])

    for key, value in tqdm(cursor, desc="Parsing LMDB"):
        try:
            datum = Datum()
            datum.ParseFromString(value)
        except DecodeError:
            print(f"Failed to decode key: {key}")
            continue

        try:
            if datum.encoded:
                img = Image.open(io.BytesIO(datum.data)).convert("RGB")
            else:
                arr = np.frombuffer(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width)
                img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
        except Exception as e:
            print(f"Image decode error: {e}")
            continue

        key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        img_filename = f"{key_str}.jpg"
        img_path = os.path.join(output_image_dir, img_filename)
        img.save(img_path)

        probs = list(datum.float_data)
        if not probs or len(probs) != 8:
            print(f"⚠️ Invalid float_data for {img_filename}")
            continue

        max_idx = int(np.argmax(probs))
        emotion_cat = emotion_names[max_idx]

        writer.writerow([
            img_filename,
            img_folder,
            emotion_cat,
            "",  # emotion_v
            "",  # emotion_a
            dataset_source
        ])

print(f"\nDone! Images saved to: {output_image_dir}, Labels saved to: {output_csv_path}")

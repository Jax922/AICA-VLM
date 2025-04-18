import os
import pandas
from tqdm import tqdm

'''
步骤1
'''
df = pandas.read_csv('./datasets/ArtEmis/artemis_dataset_release_v0.csv') # 开源标注数据
n = len(df)
ArtEmis_annotations = []
for i in tqdm(range(n), desc='Processing'):
    row = df.iloc[i]
    new_row = {
        'img_name': row['painting'],
        'img_folder': f'./datasets/ArtEmis/ArtEmis_images/{row["art_style"]}',
        'emotion_cat': row['emotion'],
        'emotion_v': '',
        'emotion_a': '',
        'emotion_reasoning': row['utterance'],
    }
    ArtEmis_annotations.append(new_row)

# 将开源标注数据进行格式规范化后保存到ArtEmis_annotations.csv文件中
ArtEmis_annotations = pandas.DataFrame(ArtEmis_annotations)
ArtEmis_annotations.to_csv('./datasets/ArtEmis/ArtEmis_annotations.csv', index=False)

'''
步骤2
'''
ArtEmis_annotations = pandas.read_csv('./datasets/ArtEmis/ArtEmis_annotations.csv')
ArtEmis_annotations.drop_duplicates(subset='img_name', inplace=True)  # 去掉重复行
n = len(ArtEmis_annotations)

for i in range(n):
    img_name = ArtEmis_annotations.iloc[i]['img_name'] + '.jpg'
    folder_path = ArtEmis_annotations.iloc[i]['img_folder']
    full_path = os.path.join(folder_path, img_name)
    if not os.path.isfile(full_path):
        print(ArtEmis_annotations.iloc[i]['img_name'])

# rembrandt_woman-standing-with-raised-hands

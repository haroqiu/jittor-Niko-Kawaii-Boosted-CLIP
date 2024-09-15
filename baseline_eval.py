import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
import config as cfg

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Animal')
parser.add_argument('--split', type=str, default='A')

args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")
classes = open('/data/classes.txt').read().splitlines()

# remove the prefix Animal, Thu-dog, Caltech-101, Food-101

new_classes = []
new_classes2 = []

for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

# training data loading
imgs_dir = '/data/'
train_labels = open('/data/train.txt').read().splitlines()
train_imgs = [imgs_dir + l.split(' ')[0] for l in train_labels]
train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
train_labels = jt.concat(train_labels)

# Evaluation on the training set
test_imgs = train_imgs
test_labels = train_labels

test_imgs = np.array(test_imgs)

top_1_list = []
top_5_list = []

eval_batch_size = cfg.EVAL_BATCH_SIZE

with jt.no_grad():
    for i in tqdm(range(0, len(test_imgs), eval_batch_size)):
        if i + eval_batch_size > len(test_imgs):
            batch_imgs = test_imgs[i:]
            batch_labels = test_labels[i:]
        else:
            batch_imgs = test_imgs[i:i + eval_batch_size]
            batch_labels = test_labels[i:i + eval_batch_size]
        images = []
        for img_path in batch_imgs:
            image = Image.open(img_path)
            image = preprocess(image)
            images.append(image)
        images = jt.stack(images)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 *
                      image_features @ text_features.transpose(0, 1)).softmax(
                          dim=-1)
        # top5 predictions
        _, top_labels = text_probs.topk(5)
        # check if the top 1 label is correct
        for j in range(len(batch_labels)):
            if top_labels[j][0] == batch_labels[j]:
                top_1_list.append(1)
            else:
                top_1_list.append(0)
            # check if the correct label is in the top 5
            if batch_labels[j] in top_labels[j]:
                top_5_list.append(1)
            else:
                top_5_list.append(0)
            
print('Top 1 Accuracy:', sum(top_1_list) / len(top_1_list))
print('Top 5 Accuracy:', sum(top_5_list) / len(top_5_list))

# dataset = 'TrainSet/' + args.dataset

# cats_dir = 'Dataset/' + dataset
# cats = os.listdir(cats_dir)

# preds = []
# top_1_list = []
# top_5_list = []

# with jt.no_grad():
#     for cat in tqdm(cats):
        
#         imgs_dir = cats_dir + '/' + cat
#         imgs = os.listdir(imgs_dir)
        
#         for img in imgs:
        
#             img_path = os.path.join(imgs_dir, img)
#             image = Image.open(img_path)
#             image = preprocess(image).unsqueeze(0)
#             image_features = model.encode_image(image)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_probs = (100.0 *
#                         image_features @ text_features.transpose(0, 1)).softmax(
#                             dim=-1)
#             # top5 predictions
#             _, top_labels = text_probs[0].topk(5)
#             preds.append(top_labels)
#             # check if the top 1 label is correct
#             if top_labels[0] == new_classes.index('a photo of ' + cat):
#                 top_1_list.append(1)
#             else:
#                 top_1_list.append(0)
#             # check if the correct label is in the top 5
#             if new_classes.index('a photo of ' + cat) in top_labels:
#                 top_5_list.append(1)
#             else:
#                 top_5_list.append(0)

# print('Dataset', args.dataset)
# print('Top 1 Accuracy:', sum(top_1_list) / len(top_1_list))
# print('Top 5 Accuracy:', sum(top_5_list) / len(top_5_list))
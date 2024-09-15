import pandas as pd
import os
import jclip as clip
import jittor as jt

jt.flags.use_cuda = 1

def get_text_features(model, prompt='prompt4'):
    df = pd.read_csv(f'/data/{prompt}.csv')
    classes_map = dict(zip(df['cat'], df['self_written_caption']))

    classes = open('/data/classes.txt').read().splitlines()
    new_classes = []

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
        new_classes.append(classes_map[c])
        
    text = clip.tokenize(new_classes)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def get_imgs():
    imgs_dir = '/data/TestSetA'
    imgs = os.listdir(imgs_dir)
    return imgs

def get_train_imgs():
    imgs_dir = '/data/'
    train_data = open('/data/selected_train3.txt').read().splitlines()
    train_imgs = [imgs_dir + l.split(' ')[0] for l in train_data]
    train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_data]
    return train_imgs, train_labels

def get_eval_imgs():
    imgs_dir = '/data/'
    eval_data = open('/data/train.txt').read().splitlines()
    train_data = open('/data/selected_train3.txt').read().splitlines()
    for l in train_data:
        if l in eval_data:
            eval_data.remove(l)
    eval_imgs = [imgs_dir + l.split(' ')[0] for l in eval_data]
    eval_labels = [jt.float32([int(l.split(' ')[1])]) for l in eval_data]
    return eval_imgs, eval_labels

def get_test_imgs():
    test_imgs = os.listdir('/data/TestSetA')
    return test_imgs

if __name__ == '__main__':
    """Test the functions"""
    model, preprocess = clip.load("/weights/ViT-B-32.pkl")
    text_features = get_text_features(model=model)
    imgs = get_imgs()
    train_imgs, train_labels = get_train_imgs()
    eval_imgs, eval_labels = get_eval_imgs()

    print(f'Number of images: {len(imgs)}')
    print(f'Number of text features: {text_features.shape[0]}')
    print(f'Number of training images: {len(train_imgs)}')
    print(f'Number of evaluation images: {len(eval_imgs)}')
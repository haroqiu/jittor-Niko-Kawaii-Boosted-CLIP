import argparse
from model import Adapted_Clip
from jittor import optim
from jittor import nn
import jclip as clip
from utls import get_text_features, get_train_imgs, get_eval_imgs, get_test_imgs
import jittor as jt
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO, # log messages that is higher or equal to {level}
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

def evaluate(eval_imgs, eval_labels, model , preprocess, text_features, eval_batch_size, alphas):
    top_1_list = []
    top_5_list = []
    
    for i in tqdm(range(0, len(eval_imgs), eval_batch_size)):
        batch_imgs = []
        batch_labels = []
        for j in range(eval_batch_size):
            if i + j >= len(eval_imgs):
                break
            img = Image.open(eval_imgs[i + j])
            img = preprocess(img).unsqueeze(0)
            batch_imgs.append(img)
            batch_labels.append(eval_labels[i + j])
        batch_imgs = jt.concat(batch_imgs, dim=0)
        batch_labels = jt.concat(batch_labels)
        logits = model.boost_execute(batch_imgs, text_features, alphas)
        _, top_labels = logits.topk(5, dim=-1)
        for j in range(len(batch_labels)):
            if top_labels[j][0] == batch_labels[j]:
                top_1_list.append(1)
            else:
                top_1_list.append(0)
            if batch_labels[j] in top_labels[j]:
                top_5_list.append(1)
            else:
                top_5_list.append(0)
                
    eval_top_1 = sum(top_1_list) / len(top_1_list)
    eval_top_5 = sum(top_5_list) / len(top_5_list)
    
    return eval_top_1, eval_top_5

def test(test_imgs, model, preprocess, text_features, test_batch_size, alphas):
    save_file = open('result.txt', 'w')
    preds = []
    imgs_dir = '/data/TestSetA/'
    
    with jt.no_grad():
        for i in tqdm(range(0, len(test_imgs), test_batch_size)):
            batch_imgs = []
            for j in range(test_batch_size):
                if i + j >= len(test_imgs):
                    break
                img = Image.open(imgs_dir + test_imgs[i + j])
                img = preprocess(img).unsqueeze(0)
                batch_imgs.append(img)
            batch_imgs = jt.concat(batch_imgs, dim=0)
            logits = model.boost_execute(batch_imgs, text_features, alphas)
            _, top_labels = logits.topk(5, dim=-1)
            preds.extend(top_labels.tolist())
            for j in range(len(batch_imgs)):
                save_file.write(test_imgs[i + j] + ' ' + ' '.join([str(p.item()) for p in top_labels[j]]) + '\n')          
            
    

def boosting_trainer(model, adapter_index, preprocess, alphas, args):
    optimizer = optim.SGD(model.adapters[adapter_index].parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_imgs, train_labels = get_train_imgs()
    text_features = get_text_features(model.clip_model)
    epoch_losses = []
    
    for epoch in range(args.num_epochs):
        batch_size = args.training_batch_size
        
        # Shuffle the images
        indexes = np.random.permutation(len(train_imgs))
        train_imgs = np.array(train_imgs)
        train_labels = np.array(train_labels)
        train_imgs = train_imgs[indexes]
        train_labels = train_labels[indexes]
        
        losses = []

        for i in range(0, len(train_imgs), batch_size):
            batch_imgs = []
            batch_labels = []
            for j in range(batch_size):
                if i + j >= len(train_imgs):
                    break
                img = Image.open(train_imgs[i + j])
                img = preprocess(img).unsqueeze(0)
                batch_imgs.append(img)
                batch_labels.append(train_labels[i + j])
            batch_imgs = jt.concat(batch_imgs, dim=0)
            batch_labels = jt.concat(batch_labels)
            logits = model.execute(batch_imgs, text_features, adapter_index)
            loss = criterion(logits, batch_labels)
            optimizer.step(loss)
            losses.append(loss[0])
        
        epoch_losses.append(jt.mean(losses).item())
        logging.info(f'Epoch: {epoch} Loss: {round(epoch_losses[-1], 3)}')
            
    # Draw the loss curve
    import matplotlib.pyplot as plt
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Adapter {adapter_index} - Loss Curve')
    plt.savefig(f'./fig/{args.adapter_type}/adapter_{adapter_index}_loss_curve.png')
    
    # Caculate the alpha
    alpha = 1
    alphas.append(alpha)
            
    if args.eval:
        print('#Evaluating the model')
        eval_imgs, eval_labels = get_eval_imgs()
        eval_batch_size = args.eval_batch_size

        with jt.no_grad():
            eval_top_1, eval_top_5 = evaluate(eval_imgs, eval_labels, model, preprocess, text_features, eval_batch_size, [1 for _ in range(adapter_index + 1)])
        print(f'Adapter {adapter_index} - Top 1 Accuracy: {eval_top_1}')
        print(f'Adapter {adapter_index} - Top 5 Accuracy: {eval_top_5}')


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_adapters', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1)
    parser.add_argument('--adapter_type', type=str, default='conv', help='mlp, attn or conv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    jt.misc.set_global_seed(args.seed)
    
    start_time = time.time()
    print('Start Time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    clip_model, preprocess = clip.load("/weights/ViT-B-32.pkl")
    model = Adapted_Clip(clip_model, args.num_adapters, adapter_type=args.adapter_type)
    
    alphas = []
    
    for i in range(args.num_adapters):
        boosting_trainer(model, i, preprocess, alphas, args)
        
    if args.test:
        print('#Testing the model')
        test_imgs = get_test_imgs()
        text_features = get_text_features(clip_model)
        test_batch_size = args.test_batch_size
        test(test_imgs, model, preprocess, text_features, test_batch_size, alphas)
        
    end_time = time.time()
    print('End Time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    print('Time Elapsed:', round(end_time - start_time, 2), 's')
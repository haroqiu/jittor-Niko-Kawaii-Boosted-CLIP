import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import glob
import numpy as np
from scipy.spatial.distance import pdist, squareform,cosine
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


jt.flags.use_cuda = 1
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()
model, preprocess = clip.load("./ViT-B-32.pkl")

imgs_dir = '/data/'
train_labels = open('/data/train.txt').read().splitlines()
train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [(int(l.split(' ')[1])) for l in train_labels]
img2lable = dict(zip(train_imgs, train_labels))

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(f"{item}\n")

img_folder_path='/data/TrainSet/'
test_folder_path='/data/TestSet/'
selected_image=[]
for sub_name in os.listdir(img_folder_path):
    for sub_name2 in os.listdir(img_folder_path+sub_name):
        # 到达具体类别folder
        image_folder=img_folder_path+sub_name+'/'+sub_name2
        image_folder = glob.glob(os.path.join(image_folder, '*.jpg'))
        # 保存图像向量
        image_path_list=[]
        image_vector_list=[]
        for image in tqdm(image_folder):
            image_path_list.append(image)
            image=Image.open(image)
            image=preprocess(image).unsqueeze(0)
            temp=model.encode_image(image)
            image_vector_list.append(list(temp.numpy()[0]))
        image_vector_list=np.array(image_vector_list)

        # 计算图像向量之间的距离矩阵
        cosine_distance_matrix = squareform(pdist(image_vector_list, 'hamming'))

        # 层次聚类
        linked = linkage(cosine_distance_matrix, 'ward')
        num_clusters = 4
        cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')
        
        # 计算每个类的中心
        cluster_centers = []
        for cluster in range(1, num_clusters + 1):
            cluster_vectors = image_vector_list[cluster_labels == cluster]
            cluster_center = cluster_vectors.mean(axis=0)
            cluster_centers.append(cluster_center)

        # 找到离每个聚类中心最近的向量
        nearest_vectors = []
        nearest_vector_indices = []
        for cluster in range(1, num_clusters + 1):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_vectors = image_vector_list[cluster_indices]
            cluster_center = cluster_centers[cluster - 1]
            distances = [cosine(vector, cluster_center) for vector in cluster_vectors]
            nearest_vector_index = np.argmin(distances)
            nearest_vector = cluster_vectors[nearest_vector_index]
            nearest_vectors.append(nearest_vector)
            nearest_vector_indices.append(cluster_indices[nearest_vector_index])
            print(f"索引: {cluster_indices[nearest_vector_index]}")


        # 选择最不确定的4张图片
        # index=np.argsort(entropy)[:4]
        print(len(nearest_vector_indices))
        # 依据index 保存图片
        for i in range(num_clusters):
            tmp_path=image_path_list[nearest_vector_indices[i]][10:]
            selected_image.append(tmp_path+' '+str(img2lable[tmp_path]))
        # break  # 测试
save_list_to_txt(selected_image, '/data/selected_train3.txt')


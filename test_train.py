
import numpy as np
import os
import pickle
from nltk.stem import PorterStemmer
import re
from pyvi import ViTokenizer
# Tạo đối tượng stemming
ps = PorterStemmer()
# Hàm để load stop words từ file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().strip().splitlines())
    return stop_words

# Đường dẫn tới file stop words của bạn
stop_words_file = 'D:/AI SIC/Project/vietnamese-stopwords-dash.txt'

# Load stop words từ file
stop_words = load_stopwords(stop_words_file)



MODEL_PATH = 'D:/MachineLearningBasic/MODEL_PATH'

def text_preprocess(text):
    # Chuyển sang chữ thường
    text = text.lower()
    
    # Loại bỏ các ký tự đặc biệt và dấu câu
    text = re.sub(r'\W', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # tách từ
    text = ViTokenizer.tokenize(text)

    # Loại bỏ stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Stemming các từ
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


# Đường dẫn đến thư mục gốc chứa các thư mục con
root_dir = 'D:/AI SIC/Project/PhanLoaiData'

label = []
# Duyệt qua từng thư mục trong thư mục gốc
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    
    # Kiểm tra nếu folder_path là một thư mục
    if os.path.isdir(folder_path):
        label.append(folder_name)  # Thêm tên thư mục vào danh sách label



# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_PATH, exist_ok=True)

with open('input.txt', 'r', encoding='utf-8') as file:
    X_test = file.read()


content = X_test

X_test = text_preprocess(X_test)
if isinstance(X_test, str):
    X_test = [X_test]
model = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
y_pred = model.predict(X_test)



# Tóm tắt nội dung


import nltk
import numpy as np
from gensim.models import KeyedVectors
from pyvi import ViTokenizer



contents_parsed = content.lower()

contents_parsed = contents_parsed.replace('\n', '. ')

contents_parsed = contents_parsed.strip()

nltk.download('punkt')
sentences = nltk.sent_tokenize(contents_parsed)

w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

X = []
for sentence in sentences:
    sentence_tokenized = ViTokenizer.tokenize(sentence)
    words = sentence_tokenized.split(" ")
    sentence_vec = np.zeros((300))
    for word in words:
        if word in w2v:
            sentence_vec += w2v[word]
    if sentence_vec.size == 1:
        sentence_vec = np.zeros((300,))
    X.append(sentence_vec)


from sklearn.cluster import KMeans

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)


from sklearn.metrics import pairwise_distances_argmin_min

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters), key=lambda k: closest[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])

y_pred_labels = [label[i] for i in y_pred]

with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(f"Chủ đề là: {', '.join(y_pred_labels)}\n")
    file.write(f"Bản tóm tắt: {summary}\n")


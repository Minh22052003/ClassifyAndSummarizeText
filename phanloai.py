from nltk.stem import PorterStemmer
import re
from pyvi import ViTokenizer
# Tạo đối tượng stemming
ps = PorterStemmer()

# tiền sử lý

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().strip().splitlines())
    return stop_words

# Đường dẫn tới file stop words
stop_words_file = 'G:/vietnamese-stopwords-dash.txt'

# Load stop words từ file
stop_words = load_stopwords(stop_words_file)

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


import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']



# Định nghĩa tỉ lệ chia tập dữ liệu
test_percent = 0.2

text = []
label = []

# Đường dẫn đến thư mục gốc chứa các thư mục con
root_dir = 'G:/PhanLoaiData'

# Duyệt qua từng thư mục trong thư mục gốc
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    
    # Kiểm tra nếu folder_path là một thư mục
    if os.path.isdir(folder_path):
        # Duyệt qua từng file .txt trong thư mục
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Phát hiện mã hóa của file
            encoding = detect_encoding(file_path)
            
            # Mở file với mã hóa đã phát hiện
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read().strip()
                content = text_preprocess(content)
                label.append(folder_name)  # Tên thư mục là nhãn
                text.append(content)  # Nội dung file là văn bản

# print(text[1])
# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)

# Mở file với mã hóa utf-8 để tránh lỗi UnicodeEncodeError
with open('train.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_train, y_train):
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w', encoding='utf-8') as fp:
    for x, y in zip(X_test, y_test):
        fp.write('{} {}\n'.format(y, x))

# Mã hóa nhãn
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# # In ra các lớp nhãn sau khi mã hóa
# print(list(label_encoder.classes_), '\n')

# Chuyển nhãn từ chuỗi sang số
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)




MODEL_PATH = 'D:/MachineLearningBasic/MODEL_PATH'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_PATH, exist_ok=True)






import pickle
import time
start_time = time.time()

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),
                                             max_df=0.8,
                                             max_features=None)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())
                    ])
text_clf = text_clf.fit(X_train, y_train)
# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))



train_time = time.time() - start_time
print('Done training Naive Bayes in', train_time, 'seconds.')


import pickle
import numpy as np
model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
y_pred = model.predict(X_test)
print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))



from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
    
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),
                                             max_df=0.8,
                                             max_features=None)), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression(solver='lbfgs',
                                                max_iter=10000,
                                                verbose=1))
                    ])
text_clf = text_clf.fit(X_train, y_train)
# save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "linear_classifier.pkl"), 'wb'))

model = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
y_pred = model.predict(X_test)
print('Linear Classifier, Accuracy =', np.mean(y_pred == y_test))

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),
                                             max_df=0.8,
                                             max_features=None)), 
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(gamma='scale'))
                    ])
text_clf = text_clf.fit(X_train, y_train)

# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))


model = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))
y_pred = model.predict(X_test)
print('SVM, Accuracy =', np.mean(y_pred == y_test))
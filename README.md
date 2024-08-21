Dự án về phân loại và tóm tắt văn bản bằng tiếng việt.

Phân loại:
+ Dự án so sánh 3 thuật toán phân loại phổ biến và đơn giản được thư viện sklearn hỗ trợ đó là Naive Bayes, Logistic Regression, Support Vector Machine.
+ Dựa theo tỉ lệ chính xác thì Logistic Regression và Support Vector Machine là những thuật toán có độ chính xác cao , Naive Bayes có độ chính xác thấp hơn nên demo test sẽ sử dụng thuật toán Logistic Regression
  
Tóm tắt:
+ Hiện dự án đang được update thêm phương thức tóm tắt chủ động nhưng do data đầu vào đào tạo khan hiếm nên chưa thể triển khai
+ Dự án sử dụng thuật toán tóm tắt thụ động bằng các phân cụm các câu có ý nghĩa gần giống nhau vào 1 cụm sử dụng thuật toán KMeans sau đó chọn lọc 1 câu quan trọng nhất trong cụm đó làm câu đại diện, gộp các câu quan trọng đó vào thì được 1 đoạn văn tóm tắt văn bản

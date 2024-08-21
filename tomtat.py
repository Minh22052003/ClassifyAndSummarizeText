import nltk
import numpy as np
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
import gensim
import os
nltk.download('punkt')

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin_min





content = """Sân chơi cho trẻ em vẫn chỉ là khẩu hiệu. Mỗi khi hè về, "Tháng hành động vì trẻ em" tới, chuyện sân chơi cho thiếu nhi lại được nhiều người quan tâm hơn. Nhưng hè nào cũng vậy, trẻ vẫn cứ thiếu chỗ chơi. Những đô thị lớn như HN và TP HCM cũng không là ngoại lệ. Thường ngày, các điểm sinh hoạt văn hoá - thể thao dành cho thiếu nhi ở HN đã luôn quá tải. Đặc biệt là ở Cung Thiếu nhi, người ta đã phải tận dụng tối đa cơ sở vật chất, huy động thêm nhiều cộng tác viên, tăng ca học cả buổi tối. Công viên nước Hồ Tây cũng là một điểm thu hút đông trẻ em tới tham gia sinh hoạt, cho dù nơi đây giá vé không phải là "mềm" và không phải các trò chơi đều phù hợp với thiếu nhi. Trong khi đó, công viên Thủ Lệ (hay còn gọi là Vườn thú HN) có diện tích rộng, nhưng lại không "bắt mắt" trẻ, bởi chuồng thú thì hôi, hàng quán choán hết lối đi. Công viên Thống Nhất có địa thế đẹp, gần trung tâm thành phố, dễ tổ chức các khu vui chơi giải trí cho trẻ, nhưng lại rất ít các trò chơi mới. Những trò như: nhà gương, đu quay. . đã nhàm chán. Trong công viên, đây đó còn xuất hiện những chợ cóc bán tạp nham đủ thứ. Cảnh "tình tự" ở nhiều công viên diễn ra vô tư trước mắt trẻ. Các bậc phu huynh rất ngại cho con mình chơi đùa, sợ giẫm phải kim tiêm của dân chích hút. Trong khi các công viên chưa đáp ứng được nhu cầu chính đáng của mọi người dân, đặc biệt là trẻ em, thì hệ thống các nhà văn hoá thiếu nhi ở HN lại rất thiếu, nặng về hình thức. Phần lớn số 1.700 điểm vui chơi cấp phường, xã trên địa bàn HN chưa được xây dựng hoàn chỉnh, hoặc để đất trống. Thậm chí ở nhiều khu dân cư, các điểm vui chơi của trẻ em đã bị thu hẹp lại hoặc bị lấn chiếm, sử dụng sai mục đích. Tại khu chung cư Thanh Xuân Bắc, đơn vị thi công đã tự ý thay đổi công năng diện tích dành cho thiếu nhi. Rạp Kim Đồng - nơi một thuở chuyên chiếu phim cho thiếu nhi - nay bị chiếm làm quán bán bia. Một vài điểm vui chơi như Sega, Star Bowl, Cosmos. . luôn có nhiều thiếu nhi tới chơi, nhưng chủ yếu là con em gia đình khá giả, bởi giá vé ở đây khá cao. Tại TPHCM hiện cũng chưa một công trình nào được xây dựng đúng nghĩa là sân chơi dành cho thiếu nhi. Nhà văn hoá thiếu nhi thành phố có mặt bằng rộng, thu hút đông bạn nhỏ, đang trở thành quá tải, nhất là trong dịp hè và lễ hội. 24 nhà văn hoá thiếu nhi quận, huyện ngoài việc có dành chỗ cho thiếu nhi sinh hoạt, thì còn cho thuê mướn mặt bằng, hoặc "tranh thủ" mở đủ loại dịch vụ cho người lớn. Trong 4 "Ngày hội tuổi thơ" dịp 1.6, Nhà văn hoá thiếu nhi thành phố có trên 15.000 lượt người đến vui chơi. Trong khi đó, vào ngày thường, nơi đây chỉ đủ đón 1.000-1.500 trẻ. Trung bình mỗi tuần có trên 20.000 lượt trẻ tập trung ở khu vực này. Cũng hằng tuần, từ các tỉnh như Bình Dương, Đồng Nai, có hàng đoàn xe chở trẻ em đến Nhà văn hoá thiếu nhi của TP HCM để sinh hoạt, học các môn năng khiếu. Học sinh các huyện Hóc Môn, Củ Chi cũng đổ về đây cuối tuần. Do không đủ chỗ, các em vẫn phải ra ngồi học ở ghế đá. Lớp học chia làm nhiều suất. Phụ huynh phải trải chiếu ngồi đợi con em mình. Thỉnh thoảng, nhà văn hoá thiếu nhi thành phố lại đón trẻ ở các mái ấm, nhà mở về vui chơi, nên càng quá tải hơn. Khi dự án mở rộng đường Nam Kỳ Khởi Nghĩa thực thi, một phần diện tích không nhỏ của nơi này sẽ bị mất đi. Sân chơi vốn đã chật sẽ càng hẹp hơn. Ban lãnh đạo Nhà văn hoá thiếu nhi đang đề xuất UBND TP cho mở địa điểm ở vùng ven, hoặc xây dựng nhà văn hoá thiếu nhi mới quy mô hơn, nhưng đến nay vẫn chưa thực hiện được. Tại quận 1, Nhà văn hoá thiếu nhi khá khang trang, nhưng hội trường chính đã trở thành. . rạp kịch của sân khấu Idecaf. Ở đây chỉ có phòng để học, chứ không có trò chơi hay khoảng sân rộng cho trẻ em vui chơi (khoảng sân này đã được trưng dụng thành bãi để xe và quán cà phê). Khu đô thị mới Phú Mỹ Hưng, dù được thiết kế phục vụ an sinh của cộng đồng, nhưng khu vui chơi của trẻ con vẫn thường bị "bỏ quên" trong dự án. Theo kiến trúc sư Võ Thành Lân, ở TPHCM chưa có công trình nào dành cho trẻ em đạt chuẩn. Nhiều dự án nhấn mạnh yếu tố làm sân chơi cho thiếu nhi, nhưng làm hay không là chuyện khác."""


# Chuyển sang chữ thường
contents_parsed = content.lower()
# Chuyển xuống dòng thành dấu chấm hoặc khoảng trắng
contents_parsed = contents_parsed.replace('\n', '. ')
# Loại bỏ khoảng trắng thừa
contents_parsed = contents_parsed.strip()
# Tách văn bản thành các câu
sentences = nltk.sent_tokenize(contents_parsed)
# Tải mô hình word2vec
model_path = "GoogleNews-vectors-negative300.bin"
w2v = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
# Chuyển đổi các câu thành vector
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
    

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)



closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters), key=lambda k: closest[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])
print(summary)







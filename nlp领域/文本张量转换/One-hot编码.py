# 导入用于保存加载对象的包
import joblib
# 导入keras中的词汇映射器Takenizer
from keras.preprocessing.text import Tokenizer

vacab = {"周杰伦", "陈奕迅", "林俊杰", "陶喆"}

# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)

# 使用映射器拟合现有文本数据
t.fit_on_texts(vacab)

for token in vacab:
    zero_list = [0] * len(vacab)
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的Onehot编码是", zero_list)

"""
林俊杰 的Onehot编码是 [1, 0, 0, 0]
陶喆 的Onehot编码是 [0, 1, 0, 0]
周杰伦 的Onehot编码是 [0, 0, 1, 0]
陈奕迅 的Onehot编码是 [0, 0, 0, 1]
"""

# 使用joblib工具保存映射器，以便之后使用
tokenizer_path = '../Tokenizer'
joblib.dump(t, tokenizer_path)


t = joblib.load("../Tokenizer")
print(type(t))
token = "林俊杰"
token_index = t.texts_to_sequences([token])[0][0] - 1
zero_list = [0] * 5
zero_list[token_index] = 1
print(zero_list)

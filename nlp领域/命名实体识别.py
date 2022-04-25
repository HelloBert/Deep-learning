# 命名实体识别
import hanlp

recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)

# 中文命名实体识别，用hanlp工具输入必须是用列表切分的单个字符
print(recognizer(list("上海华安工业（集团）公司董事长谭旭光和秘书长张晚霞来到美国纽约现代艺术博物馆参观")))

# 英文命名实体识别
recognizer1 = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_CASED_EN)
print(recognizer1(["President", "Obama", "is", "speaking", "at", "the", "white", "House"]))
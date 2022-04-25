import jieba.posseg as pseg
import hanlp

print(pseg.lcut('我爱北京天安门'))

# [pair('我', 'r'), pair('爱', 'v'), pair('北京', 'ns'), pair('天安门', 'ns')]

tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
print(tagger(["我", "的", "希望", "是", "希望", "和平"]))
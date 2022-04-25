import jieba
import hanlp

# jieba.cutl如果不给参数，默认是精确模式
content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
# jieba.cut(content, cut_all=False)
# 返回一个生成器对象,cut_all=False是精确模式
print(jieba.lcut(content, cut_all=False))

# cut_all=True 全模式，里面所有能作为词的汉字组合全被提取出来，全模式里面的词有一些并不是必须要的。
print(jieba.lcut(content, cut_all=True))

# 搜索引擎模式，相对于全模式，只会切分比较长的词
jieba.cut_for_search(content)
print(jieba.lcut_for_search(content))


tokenizer = hanlp.load('CTB6_CONVSEG')
print(tokenizer("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"))
tokenizer=hanlp.utils.rules.tokenizer_english
print(tokenizer("Open your books and turn to page 20"))


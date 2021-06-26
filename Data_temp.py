import jieba
from hanziconv import HanziConv
import re
datas = []
# lines = open('data/qa1.txt','r',encoding='utf-8').readlines()
# print(len(lines))
def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]
for line in open('data/qa1.txt','r',encoding='utf-8').readlines():
    if not line:
        continue
    data_list = line.strip().split('|')
    if len(data_list)<2:
        continue
    else:
        datas.append(list(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]','',data_list[-1]))))
    # if not datas:
    #     datas.append(data_list[1])
    # while datas and datas[-1] == data_list[1]:
    #     datas.pop(-1)
    # datas.append(dre.sub(r'[^\u4e00-\u9fa5]','','ABC中，。*')ata_list[1])
print(len(datas))

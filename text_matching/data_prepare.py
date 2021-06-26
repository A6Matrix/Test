import re
import jieba
import random
from tensorflow.contrib import learn


class Data_Prepare(object):

    def readfile(self, filename):
        texta = []
        textb = []
        tag = []
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                if len(line) != 3:
                    continue
                texta.append(self.pre_processing(line[0]))
                textb.append(self.pre_processing(line[1]))
                tag.append(line[2])

        index = [x for x in range(len(texta))]
        random.shuffle(index)
        texta_new = [texta[x] for x in index]
        textb_new = [textb[x] for x in index]
        tag_new = [int(tag[x]) for x in index]

        type = list(set(tag_new))
        dicts = {}
        tags_vec = []
        for x in tag_new:
            if x not in dicts.keys():
                dicts[x] = 1
            else:
                dicts[x] += 1
            temp = [0] * len(type)
            temp[int(x)] = 1
            tags_vec.append(temp)
        return texta_new, textb_new, tags_vec


    # def read_pre_file(self,filename):
    #     texta = []
    #     textb = []
    #     with open(filename, 'r',encoding='utf-8') as f:
    #         for line in f.readlines():
    #             line = line.strip().split("\t")
    #             texta.append(self.pre_processing(line[0]))
    #             textb.append(self.pre_processing(line[1]))
    #     return texta,textb

    def read_pre_file(self,filename):
        text = []
        # textb = []
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                text.append(self.pre_processing(line))
        return text

    def pre_processing(self, text):
        text = text.lower()
        # 删除（）里的内容
        text = re.sub('[,?"#();:!…？“”]', '', text)
        text = re.sub('[/]', ' ', text)
        text = re.sub(r"can't", 'can not', text)
        text = re.sub(r"n't", ' not', text)
        text = re.sub(r"'ve", " 've", text)
        text = re.sub(r"'d", " 'd", text)
        text = re.sub(r"'ll", " 'll", text)
        text = re.sub(r"'m", " 'm", text)
        text = re.sub(r"'s", " 's", text)

        return text

    def build_vocab(self, sentences, path):
        lens = [len(sentence.split(" ")) for sentence in sentences]
        max_length = max(lens)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
        vocab_processor.fit(sentences)
        vocab_processor.save(path)


if __name__ == '__main__':
    data_pre = Data_Prepare()
    data_pre.read_pre_file('data/ENG_sim_test.txt')

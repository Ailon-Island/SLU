import json
import pycorrector

datas = json.load(open('data/development.json', 'r', encoding='utf-8'))
for data in datas:
    for utt in data:
        utt['asr_1best'], w = pycorrector.correct(utt['asr_1best'])
        print(utt['asr_1best'], w)

json.dump(datas, open('data/development_corrected.json', 'w', encoding='utf-8'), ensure_ascii=False)
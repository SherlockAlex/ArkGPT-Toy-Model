import json
import random
# 小说
novel_files_count = 138
english_files_count = 44
wiki_files_count = 1

def train_data():

    data = {
        'english':[],
        'novel':[],
        'wiki':[],
        'words':[]
    }

    for file in range(english_files_count-10):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            data['english'].append({'text':f.read()})

    for file in range(novel_files_count-50):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text':f.read()})
    
    for file in range(wiki_files_count):
        with open('./语料/百科/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['wiki'].append({'text':f.read()})

    with open('./语料/词典/ci.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)

    with open('./语料/词典/idiom.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)

    with open('./语料/词典/word.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)

    with open('./语料/词典/xiehouyu.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)

    for key in data.keys():
        random.shuffle(data[key])
    
    data = str(data)
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')

    return data

def valid_data():
    data = {
        'english':[],
        'novel':[],
    }

    for file in range(english_files_count-4,english_files_count):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            data['english'].append({'text':f.read()})

    for file in range(novel_files_count-5,novel_files_count):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text':f.read()})

    for key in data.keys():
        random.shuffle(data[key])

    data = str(data)
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')

    return data

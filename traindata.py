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

    for file in range(english_files_count-4):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            data['english'].append({'text':f.read()})
        f.close()

    for file in range(novel_files_count-5):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text':f.read()})
        f.close()
    
    for file in range(wiki_files_count):
        with open('./语料/百科/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['wiki'].append({'text':f.read()})
        f.close()

    with open('./语料/词典/ci.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)
    f.close()

    with open('./语料/词典/idiom.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)
    f.close()

    with open('./语料/词典/word.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)
    f.close()

    with open('./语料/词典/xiehouyu.json',encoding='utf-8') as f:
        ci = json.load(f)
        data['words'].append(ci)
    f.close()

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
        f.close()

    for file in range(novel_files_count-5,novel_files_count):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text':f.read()})
        f.close()

    for key in data.keys():
        random.shuffle(data[key])

    data = str(data)
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')

    return data

def news_data(begin:int,last:int):
    data = {
        'news':[]
    }

    for i in range(begin,last):
        with open(f'./语料/人民日报/{i}.txt','r',encoding='utf-16') as f:
            data['news'].append({"year":i,"text":f.read()})
        f.close()
        
    data = str(data)
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')
    data = data.replace('\\xa0','')

    return data

def dialog_train():
    data = ''

    for i in range(1,10):
        with open(f'./语料/datasets/pCLUE_train_{i}.json','r',encoding='utf-8') as f:
            data = data + f.read()
        f.close()
        pass
    
    
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')
    return data
    pass

def dialog_test():
    data = ''

    for i in range(1,3):
        with open(f'./语料/datasets/pCLUE_test_{i}.json','r',encoding='utf-8') as f:
            data = data + f.read()
        f.close()
        pass
    
    
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')
    return data
    pass

def dataset(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    return data["prompt"]
    pass


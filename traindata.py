import json
import random
import gc
# 小说
novel_files_count = 138
english_files_count = 44
wiki_files_count = 1

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


def prompt_dataset(filepath):
    '''
    json文件应该为如下格式\n
    {
        "prompt":\n
        [
            你的prompt
        ]
    }
    '''
    with open(filepath,'r',encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    return data["prompt"]
    pass

def pretrain_train_data():

    data = {
        'english':[],
        'novel':[],
        'wiki':[],
        'words':[],
        'news':[],
        'dailog':[],
    }

    for file in range(english_files_count-4):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            data['english'].append({'text'+str(file):f.read()})
        f.close()

    for file in range(novel_files_count-5):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text'+str(file):f.read()})
        f.close()
    
    for file in range(wiki_files_count):
        with open('./语料/百科/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['wiki'].append({'text'+str(file):f.read()})
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
    print("加载预训练训练集成功")
    gc.collect()

    return data

def pretrain_valid_data():
    data = {
        'english':[],
        'novel':[],
        'news':[],
        'dailog':[],
        'wiki':[]
    }

    for file in range(english_files_count-4,english_files_count):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            data['english'].append({'text':f.read()})
        f.close()
    gc.collect()

    

    for file in range(novel_files_count-5,novel_files_count):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            data['novel'].append({'text':f.read()})
        f.close()
    gc.collect()

    for key in data.keys():
        random.shuffle(data[key])
    gc.collect()

    data = str(data)
    data = data.replace('\\n','\n')
    data = data.replace('\\u3000',' ')
    gc.collect()

    print("加载预训练测试集集成功")
    gc.collect()

    return data




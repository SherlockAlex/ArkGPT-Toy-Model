import json

# 小说
novel_files_count = 138
english_files_count = 44

def get_text():
    text = ""

    for file in range(english_files_count):
        with open('./语料/英语/'+str(file)+'.txt', 'r',encoding="utf-8") as f:
            text = text + f.read()

    for file in range(novel_files_count):
        with open('./语料/小说/'+str(file)+'.txt', 'r',encoding="utf-16") as f:
            text = text + f.read()
    
    return text

def get_size():
    text = get_text()

    memory_size = len(text)/1024/1024
    print(f"小说数量:{len(novel_files_count+english_files_count)},字数:{len(text)},大小:{memory_size} MB") #1MB


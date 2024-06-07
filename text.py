import json

science_files=[
    './量子力学史话.txt',
    './本草纲目.txt',
    './大语言模型.txt',
    './量子力学笔记.txt'
]

# 小说
novel_files = [
    './百科全书学者.txt',
    './史记.txt',
    './孙子兵法.txt',
    './左传.txt',
    './三国志.txt',
    './三国演义.txt',
    './红楼梦.txt',
    './西游记.txt',
    './诛仙.txt',
    './量子力学史话.txt',
    './射雕英雄传.txt',
    './神雕侠侣.txt',
    './倚天屠龙记.txt',
    './天龙八部.txt',
    './剑来.txt',
    './钢铁侠.txt',
    './鲁滨逊漂流记.txt',
    './格林童话.txt',
    './三体.txt',
    './斗破苍穹.txt',
    './海底两万里.txt',
    './基地.txt',
    './阿西莫夫科幻圣经.txt',
    './平凡的世界.txt',
    './战争与和平.txt',
    './笑傲江湖.txt',
    './侠客行.txt',
    './碧血剑.txt',
    './吞噬星空.txt',
    './庆余年.txt',
    './朱雀记.txt',
    './从一到无穷大.txt',
    './福尔摩斯探案集.txt',
    './大军师联盟.txt',
    './一念永恒.txt',
    './北洋军阀史话.txt',
    './山海经.txt',
    './微微一笑很倾城.txt',
    './校花的贴身高手.txt',
    './一数封神.txt',
    './仙人消失之后.txt',
    './白骨大圣.txt'
]

def get_text():
    text = ""

    for file in novel_files:
        with open('./语料/小说/'+file, 'r',encoding="utf-16") as f:
            text = text + f.read()

    for file in science_files:
        with open('./语料/科学/'+file, 'r',encoding="utf-16") as f:
            text = text + f.read()

    #text = text.replace("\n","")
    text = text.replace("※","")
    #text = text.replace("　","")
    text = text.replace("*","")
    #text = text.replace(" ","")
    return text

def get_size():
    text = get_text()

    memory_size = len(text)/1024/1024
    print(f"小说数量:{len(novel_files)},字数:{len(text)},大小:{memory_size} MB") #1MB


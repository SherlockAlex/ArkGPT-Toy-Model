import tiktoken
import traindata

set = traindata.prompt_dataset('./语料/wiki_train.json')

print(max(set,key=len))

coder = tiktoken.get_encoding("cl100k_base")

b = coder.encode('"input":"你叫什么名字？"}<EOS>')
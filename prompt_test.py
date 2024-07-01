import tiktoken

coder = tiktoken.get_encoding("cl100k_base")

b = coder.encode('"input":"你叫什么名字？"}<EOS>')
print(len(b))
print(len(b[:-1]))
print(len(b[1:]))
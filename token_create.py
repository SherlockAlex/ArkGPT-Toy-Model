import tiktoken

tokener = tiktoken.get_encoding('cl100k_base')
text = 'Hello what is your name?My name is Javis!Nice to meet you,sir.'
print(tokener.decode([12870, 94]),tokener.encode(text))
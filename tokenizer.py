import json
import jieba

class Tokenizer():
    def __init__(self,mode="words",place_holder="#",encoding = 'utf-8'):
        self.word_to_token = {}
        self.token_to_word = {}
        self.encoding = encoding
        self.place_holder = place_holder
        self.mode = mode
        pass

    def read(self,filename):
        self.file_path = filename
        return self
        pass

    def slide(self,text):
        if self.mode == "words":
            words = jieba.cut(text)
        else:
            words = text
        return words
        pass

    def create(self):
        with open(self.file_path,"r",encoding=self.encoding) as file:
            text = file.read()
        words = self.slide(text)
        self.word_to_token = {word:token+2 for token,word in enumerate(set(words))}
        self.token_to_word = {token:word for word,token in self.word_to_token.items()}
        return self
        pass

    def get_vocab_count(self):
        vocab_len =  len(self.word_to_token)+2
        return vocab_len
        pass

    def get_token(self,word):
        if word not in self.word_to_token.keys():
            return 1
        return self.word_to_token[word]
        pass
    
    def encode(self,sentence,max_token = None,use_begin_placeholder = False):

        words = self.slide(sentence)
        sentence_words = list(words)
        tokens = []
        if max_token == None:
            for word in sentence_words:
                token = self.word_to_token[word]
                tokens.append(token)
                pass
            if use_begin_placeholder:
                tokens = [1]+tokens
            return tokens
            pass

        if sentence == "" or sentence == self.place_holder:
            return [0 for i in range(max_token)]
        
        
        sentence_len = len(sentence_words)
        tokens = [0 for i in range(max_token)]

        if sentence_len>max_token:
            sentence_len = max_token
            sentence_words = sentence_words[-max_token:]
        
        for i in range(sentence_len):
            word = sentence_words[i]
            if word not in self.word_to_token.keys():
                tokens[i] = 1
                continue
                pass
            token = self.word_to_token[word]
            tokens[i] = token
            pass
        
        if use_begin_placeholder:
            tokens = [1] + tokens
            tokens = tokens[0:max_token]
        return tokens
    pass

    def get_word(self,token):
        query = int(token)
        if query in self.token_to_word:
            return self.token_to_word[query]
        pass

    def decode(self,tokens,use_stop_placeholder = True):
        words = []
        for token in tokens:
            if token == 0 and use_stop_placeholder:
                break
            
            if token == 0 and not use_stop_placeholder:
                continue
            if token == 1:
                continue
            if token in self.token_to_word:
                words.append(self.token_to_word[token])
        sentence = "".join(words)
        return sentence
        pass
    
    def save(self, filename):
        with open(filename, 'w',encoding=self.encoding) as file:
            json.dump(self.word_to_token, file)

    def load(self, filename):
        with open(filename, 'r',encoding=self.encoding) as file:
            self.word_to_token = json.load(file)
            self.token_to_word = {token:word for word,token in self.word_to_token.items()}
        return self

    def head(self):
        return self.token_to_word
        pass
    
    pass
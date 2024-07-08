from ContextManager import ContextManager
import torch
import torch.nn as nn
import traindata

import torch

model = ContextManager().load()

with open("./神雕侠侣.txt","r",encoding="utf-8") as f:
    text =f.read()

database = model.create_database(text)
result = model.search("杨过断了一条手臂",database)
print(result)
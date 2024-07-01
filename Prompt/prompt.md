# prompt类型

* Generate：生成型prompt期望输出是创造性的，例如生成一段文本、一个故事或一个答案。即使有多种回答，只要它们都是创造性的，并且符合prompt的要求，它们仍然可以归类为生成型。

* NLI (Natural Language Inference)：自然语言推理型prompt期望输出是对给定文本对之间逻辑关系的判断，如蕴含、矛盾或中立。

* MRS (Machine Reading Comprehension)：机器阅读理解型prompt期望输出是针对特定问题的答案，通常需要从给定的文本中提取信息。

* Classify：分类型prompt期望输出是对输入文本的分类，例如情感分析、主题分类等。

# 引导
请你编写几条微调数据集，服从下面格式：{"input":"","target":"","answer_choices":[],"type":""}，其中，type的取值有：generate、nli、mrs、classify，而"answer_choices"项表示其他答案，这一项可以要，也可以不要，不要的时候prompt就不出现"answer_choices"

10

30
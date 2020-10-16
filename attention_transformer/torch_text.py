# 1. specify now preprocessing should be done -> fields
# 2. Use Dataset to load the data -> TabularDataset (JSON, CSV, TSV)
# 3. Construct an iterator to do batching and padding -> BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator

tokenize = lambda x: x.split()

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'quote': ('q', quote), 'score': ('s', score)}

train_data, test_data = TabularDataset.splits(
                                            path='dataset/torchtext_test_dataset',
                                            train='train.json',
                                            test='test.json',
                                            # validation = 'validation.json'
                                            format='json',
                                            fields=fields
                                        )

# train_data, test_data = TabularDataset.splits(
#                                             path='dataset/torchtext_test_dataset',
#                                             train='train.csv',
#                                             test='test.csv',
#                                             format='csv',
#                                             fields=fields
#                                         )
#
# train_data, test_data = TabularDataset.splits(
#                                             path='dataset/torchtext_test_dataset',
#                                             train='train.tsv',
#                                             test='test.tsv',
#                                             format='tsv',
#                                             fields=fields
#                                         )

print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())


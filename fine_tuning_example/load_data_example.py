import itertools
import json
import time

from datasets import load_dataset

# Common Crawl's web crawl corpus (C4) is a dataset that contains a cleaned version of the Common Crawl web corpus.
# load subset "en", split "train" of the dataset, and use streaming=True to load the dataset as a stream.
start_time = time.time()
pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
end_time = time.time()
print("Time taken to load the dataset as a stream: ", end_time - start_time, "seconds")

start_time = time.time()
n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(json.dumps(i, indent=4))

end_time = time.time()
print("Time taken to print top", n, "items: ", end_time - start_time, "seconds")
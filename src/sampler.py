import random
from collections import defaultdict
from torch.utils.data import Sampler

class ClassAwareSampler(Sampler):
    def __init__(self, dataset, classes_per_batch=8, samples_per_class=4):
        self.dataset = dataset
        self.cpb = classes_per_batch
        self.spc = samples_per_class
        buckets = defaultdict(list)  
        for i in range(len(dataset)):
            _, _, y = dataset[i]
            if y >= 0: buckets[int(y)].append(i) 
        self.buckets = {k:v for k,v in buckets.items() if v}
        self.classes = list(self.buckets.keys())

    def __iter__(self):
        while len(self.classes) >= self.cpb:
            chosen = random.sample(self.classes, self.cpb)
            batch = []
            for c in chosen:
                pool = self.buckets[c]
                inds = random.sample(pool, self.spc) if len(pool) >= self.spc else random.choices(pool, k=self.spc)
                batch.extend(inds)
            
            yield batch
    def __len__(self):
        total = sum(len(v) for v in self.buckets.values())
        return total // (self.cpb * self.spc)
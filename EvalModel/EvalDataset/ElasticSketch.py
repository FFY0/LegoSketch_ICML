import probables.countminsketch as cm


class ElasticSketch():
    def __init__(self, mem_kb, unit_budget):
        unit_budget = int(unit_budget)
        self.light_part_mem = int(((mem_kb * 4) / 5) // unit_budget) * unit_budget
        if self.light_part_mem < unit_budget:
            self.light_part_mem = unit_budget
        self.heavy_part_mem = mem_kb - self.light_part_mem
        assert self.heavy_part_mem > 0
        self.bucket_num = (self.heavy_part_mem * 1024) // 9
        self.heavy_part = HeavyPart(self.bucket_num)
        self.light_part = LightPart(self.light_part_mem * 1024)
        print('ElasticSketch: heavy_part_mem ', self.heavy_part_mem, 'light_part_mem ', self.light_part_mem)

    def insert(self, key, value):
        key = hash(key)
        res = self.heavy_part.insert(key, value)
        flag = res[0]
        if flag == 3:
            self.light_part.insert(key, value)
        elif flag == 4:
            evict_key = res[1]
            evict_value = res[2]
            self.light_part.insert(evict_key, evict_value)
        return res

    def query(self, key):
        key = hash(key)
        heavy_count, Flag = self.heavy_part.query(key)
        if Flag is True or heavy_count is None:
            light_count = self.light_part.query(key)
            if heavy_count is None:
                assert light_count is not None
                return light_count, heavy_count, Flag
            else:
                assert light_count is not None and heavy_count is not None
                return light_count + heavy_count, heavy_count, Flag
        else:
            return heavy_count, heavy_count, Flag


class HeavyPart():
    def __init__(self, bucket_num, lambda_param=10):
        self.bucket_num = bucket_num
        self.bucket_list = []
        self.lambda_param = lambda_param
        for i in range(bucket_num):
            self.bucket_list.append((None, 0, False, 0))

    def insert(self, key, value):
        pos = key % self.bucket_num
        bucket = self.bucket_list[pos]
        if bucket[0] is None:
            self.bucket_list[pos] = (key, value, False, 0)
            return 1, None, None
        elif bucket[0] == key:
            self.bucket_list[pos] = (bucket[0], bucket[1] + value, bucket[2], bucket[3])
            return 2, None, None
        elif bucket[3] / bucket[1] < self.lambda_param:
            self.bucket_list[pos] = (bucket[0], bucket[1], bucket[2], bucket[3] + value)
            # need to insert into cm
            return 3, None, None
        elif bucket[3] / bucket[1] >= self.lambda_param:
            self.bucket_list[pos] = (key, value, True, 1)
            return 4, bucket[0], bucket[1]

    def query(self, key):
        pos = key % self.bucket_num
        bucket = self.bucket_list[pos]
        if bucket[0] is None or bucket[0] != key:
            return None, True
        elif bucket[0] == key:
            return bucket[1], bucket[2]


class LightPart():
    def __init__(self, mem):
        depth = 3
        width = mem // (depth * 4)
        self.cm = cm.CountMinSketch(width=width, depth=depth)

    def insert(self, key, value):
        assert key is not None
        self.cm.add(str(key), value)

    def query(self, key):
        return self.cm.check(str(key))

    def reset(self, mem):
        depth = 4
        width = mem // (depth * 4)
        self.cm = cm.CountMinSketch(width=width, depth=depth)

    def clear(self):
        self.cm.clear()

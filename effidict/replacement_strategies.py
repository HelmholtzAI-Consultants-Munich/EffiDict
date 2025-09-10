from collections import OrderedDict, defaultdict
import random
from abc import abstractmethod

class ReplacementStrategy:
    def __init__(self, disk_backend, max_in_memory):
        self.memory = self.get_memory()
        self.disk_backend = disk_backend
        self.max_in_memory = max_in_memory

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def put(self, key, value):
        pass
    
    @abstractmethod
    def get_memory(self):
        pass
        
    
    # --- NEW: Method for handling deletion from memory ---
    def delete(self, key):
        """Remove a key from internal memory structures."""
        if key in self.memory:
            del self.memory[key]

    def clear(self):
        """Clear internal memory structures."""
        self.memory.clear()

class RandomReplacement(ReplacementStrategy):
    def get(self, key):
        if key in self.memory:
            return self.memory[key]
        else:
            return self.disk_backend.deserialize(key)

    def put(self, key, value):
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            random_key = random.choice(list(self.memory.keys()))
            random_value = self.memory.pop(random_key)
            self.disk_backend.serialize(random_key, random_value)

    def get_memory(self):
        return defaultdict()
    
class FIFOReplacement(ReplacementStrategy):
    def get(self, key):
        if key in self.memory:
            return self.memory[key]
        else:
            return self.disk_backend.deserialize(key)
        
    def put(self, key, value):
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem(last=False)
            self.disk_backend.serialize(oldest_key, oldest_value)

    def get_memory(self):
        return OrderedDict()
    
class LIFOReplacement(ReplacementStrategy):
    def get(self, key):
        if key in self.memory:
            return self.memory[key]
        else:
            return self.disk_backend.deserialize(key)
        
    def put(self, key, value):
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem()
            self.disk_backend.serialize(oldest_key, oldest_value)

    def get_memory(self):
        return OrderedDict()
    
class LRUReplacement(ReplacementStrategy):
    def get(self, key):
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        else:
            value = self.disk_backend.deserialize(key)
            if value is not None:
                self.put(key, value)
            return value
        
    def put(self, key, value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem(last=False)
            self.disk_backend.serialize(oldest_key, oldest_value)

    def get_memory(self):
        return OrderedDict()
    
class MRUReplacement(ReplacementStrategy):
    def get(self, key):
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        else:
            value = self.disk_backend.deserialize(key)
            if value is not None:
                self.put(key, value)
            return value
        
    def put(self, key, value):
        self.memory[key] = value
        self.memory.move_to_end(key)
        if len(self.memory) > self.max_in_memory:
            oldest_key, oldest_value = self.memory.popitem()
            self.disk_backend.serialize(oldest_key, oldest_value)

    def get_memory(self):
        return OrderedDict()
    
class LFUReplacement(ReplacementStrategy):
    def __init__(self, disk_backend, max_in_memory):
        super().__init__(disk_backend, max_in_memory)
        self.secondary_memory = defaultdict(int)

    def get(self, key):
        if key in self.memory:
            self.secondary_memory[key] += 1
            return self.memory[key]
        else:
            value = self.disk_backend.deserialize(key)
            self.put(key, value) # simplified: put handles loading
            return value
        
    def put(self, key, value):
        # Increment frequency if key exists, otherwise set to 1
        self.secondary_memory[key] = self.secondary_memory.get(key, 0) + 1
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            # Find the key with the minimum frequency to evict
            # Exclude the key we just added from eviction candidates
            eviction_candidates = {k: v for k, v in self.secondary_memory.items() if k != key}
            if eviction_candidates:
                min_key = min(eviction_candidates, key=eviction_candidates.get)
                min_value = self.memory.pop(min_key)
                self.disk_backend.serialize(min_key, min_value)
                del self.secondary_memory[min_key]

    def get_memory(self):
        return {} # Plain dict is fine here
    
    # --- OVERRIDE: Also clean up the frequency counter ---
    def delete(self, key):
        super().delete(key)
        if key in self.secondary_memory:
            del self.secondary_memory[key]

    def clear(self):
        super().clear()
        self.secondary_memory.clear()

# Similar changes for MFUReplacement...
class MFUReplacement(ReplacementStrategy):
    def __init__(self, disk_backend, max_in_memory):
        super().__init__(disk_backend, max_in_memory)
        self.secondary_memory = defaultdict(int)

    def get(self, key):
        if key in self.memory:
            self.secondary_memory[key] += 1
            return self.memory[key]
        else:
            value = self.disk_backend.deserialize(key)
            self.put(key, value)
            return value
        
    def put(self, key, value):
        self.secondary_memory[key] = self.secondary_memory.get(key, 0) + 1
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            eviction_candidates = {k: v for k, v in self.secondary_memory.items() if k != key}
            if eviction_candidates:
                max_key = max(eviction_candidates, key=eviction_candidates.get)
                max_value = self.memory.pop(max_key)
                self.disk_backend.serialize(max_key, max_value)
                del self.secondary_memory[max_key]

    def get_memory(self):
        return {}
    
    def delete(self, key):
        super().delete(key)
        if key in self.secondary_memory:
            del self.secondary_memory[key]

    def clear(self):
        super().clear()
        self.secondary_memory.clear()


    def __init__(self, disk_backend, max_in_memory):
        super().__init__(disk_backend, max_in_memory)
        self.secondary_memory = defaultdict(int)

    def get(self, key):
        if key in self.memory:
            self.secondary_memory[key] += 1
            return self.memory[key]
        else:
            value = self.disk_backend.deserialize(key)
            if value is not None:
                self.put(key, value)
            return value
        
    def put(self, key, value):
        self.secondary_memory[key] = 1
        self.memory[key] = value
        if len(self.memory) > self.max_in_memory:
            max_key = max(self.secondary_memory, key=self.secondary_memory.get)
            max_value = self.memory.pop(max_key)
            self.disk_backend.serialize(max_key, max_value)
            self.secondary_memory.pop(max_key)

    def get_memory(self):
        return OrderedDict()
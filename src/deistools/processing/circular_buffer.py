import numpy as np
from attrs import define, field

@define
class NumpyCircularBuffer:
    maxlen : int
    _head : int = 0
    _tail: int = 0
    _data : np.array = field() 
    @_data.default
    def _initialize_buffer(self):
        return  np.zeros(self.maxlen, dtype = np.int32)

    def push(self, new_data:np.array):
        '''
        Append data to the buffer, overwrite on overflow is allowed.
        '''
        if new_data.size > self.maxlen:
            raise Exception('Data input is longer then buffer size. Operation not possible.')
        
        if self._tail + new_data.size > self.maxlen: 
            overflow = (self._tail + new_data.size) - self.maxlen
            self._data[self._tail:self.maxlen] = new_data[:self.maxlen - self._tail]
            self._data[0:overflow] = new_data[self.maxlen - self._tail:]
            self._tail = overflow 
            self._head = overflow
        else:
            self._data[self._tail:self._tail+new_data.size] = new_data
            self._tail = self._tail + new_data.size
            if self._tail > self._head:
                self._head = self._tail


    def pop(self, num_elements:int):
        '''
        Return a desired number of elements from the buffer.
        '''
        if num_elements > self.maxlen:
            raise Exception('Elements requested are more then buffer size. Operation not possible.')
        
        if self._head + num_elements>self.maxlen:
            overflow = (self._head + num_elements) - self.maxlen
            current_head = self._head
            self._head = overflow
            return np.concatenate(
                (self._data[current_head:self.maxlen], 
                self._data[0:overflow] 
                ),
                axis=0,
                )
        else:
            self._head = self._head + num_elements
            return self._data[self._head - num_elements:self._head].copy()

    
    def get_data(self):
        '''
        Get data from head to tail withot removing them from the buffer.
        '''
        if self._tail <= self._head:
            return np.concatenate(
                (self._data[self._head:], 
                self._data[:self._tail] 
                ),
                axis=0,
                )
        else:
            return self._data[self._head:self._tail]
        
    def print_status(self):
        print(f'Maximum lenght: {self.maxlen}')
        print(f'Head: {self._head}')
        print(f'Tail: {self._tail}')
        print(f'Elements: {self._data}')
        
    def get_lenght(self):
        if self._tail <= self._head:
            return(self.maxlen - self._head) + self._tail
        else:    
            return self._tail - self._head 

if __name__ == "__main__":
    b = NumpyCircularBuffer(10)
    b.push(np.array([1,2,3,4,5,6,7,8,9,10]))
    print(b.get_lenght())
    b.print_status()
    b.push(np.array([11]))
    b.print_status()
    b.push(np.array([12,13,14,15,16,17]))
    print(b.get_lenght())
    b.print_status()
    print(f'Buffered data: {b.get_data()}')
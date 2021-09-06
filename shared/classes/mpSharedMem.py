#############################################################################
# wrapper for multiprocessing pool for easier setup of shared arrays
# required: Python >= 3.8
#############################################################################

import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from multiprocessing import shared_memory

class SharedArray:
	"""
	create a shared array object for multiprocessing processes
	attributes: shape, dtype, sharedMem
    Example Usage:
        In main thread, do
            sharedArr = SharedArray(arr) where arr is np.array
        In child threads, pass sharedArr to kernel function and do
            arr_ = sharedArr.getSharedArray()
        After parallel processing, in main thread, use
            sharedArr.getSharedArrayValue() to get a read-only copy the array
            sharedArr.getSharedArray() to get the real array, writing to this array will change the shared location
        When done, do
            sharedArr.destroy() to avoid memory leaks
	"""
	def __init__(self, arr):
		self.shape = arr.shape
		self.dtype = arr.dtype
		# init a new numpy array on shared memory buffer
		sharedMemSize = np.dtype(arr.dtype).itemsize * np.prod(arr.shape)
		print('creating shared memory with size: ' + str(sharedMemSize), flush = True)
		self.sharedMem = shared_memory.SharedMemory(create = True, size = sharedMemSize)
		# copy data to shared memory
		print('creating numpy array from shared mem ... ' , flush = True)
		sharedArray = np.ndarray(shape = arr.shape, dtype = arr.dtype, buffer = self.sharedMem.buf)
		print('copying data to shared mem ... ' , flush = True)
		sharedArray[:] = arr[:]
		print('done')

	def getSharedArray(self):
		"""
		return an np.array instance of the shared array. writing to this copy will change the shared array
		"""
		sharedMem = shared_memory.SharedMemory(name = self.sharedMem.name)
		# create numpy object from existing memory buffer
		arr = np.ndarray(shape = self.shape, dtype = self.dtype, buffer = self.sharedMem.buf)
		return arr

	def getSharedArrayValue(self):
		"""
		return a copy of the shared array. writing to this copy doesn't affect the shared array
		"""
		arr = self.getSharedArray()
		# create output array and copy data over
		outArr = np.ndarray(shape = arr.shape, dtype = arr.dtype)
		outArr[:] = arr[:]
		return outArr

	def destroy(self):
		""" free memory """
		self.sharedMem.close()  # close after using
		self.sharedMem.unlink()  # free memory


# kernel function to test read sharedArr
def test_read_kernel(sharedArr_):
	arr_ = sharedArr_.getSharedArray()
	shape = sharedArr_.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			if arr_[i, j] != 0:
				return 1 # failed test
	return 0 # pass test

def test_read():
	""" create a shared array and fill with 0 """
	DIM1 = 3; DIM2 = 115000
	arr = np.zeros([DIM1, DIM2])
	sharedArr = SharedArray(arr)
	p = mp.Pool(32)
	args = [sharedArr]*32
	result = p.map(test_read_kernel, args)
	if sum(result)==0:
		print('test read passed')
	else:
		print('test read failed')
	sharedArr.destroy()

# kernel function to test write sharedArr
def test_write_kernel(args):
	sharedArr_, i, j = args
	arr_ = sharedArr_.getSharedArray()
	shape = sharedArr_.shape
	arr_[i,j] = 1

def test_write():
	""" create a shared array and fill with 1 """
	arr = np.zeros([3, 115000], dtype = 'float32')
	sharedArr = SharedArray(arr)
	args = []
	for i in range(3):
		for j in range(115000):
			args.append((sharedArr, i, j))
	p = mp.Pool(32)
	p.map(test_write_kernel, args)
	new_arr = sharedArr.getSharedArrayValue()
	for i in range(3):
		for j in range(115000):
			if new_arr[i,j] != 1:
				print('test write failed')
				return
	print('test write passed')
	sharedArr.destroy()

if __name__ == '__main__':
	test_read()
	test_write()


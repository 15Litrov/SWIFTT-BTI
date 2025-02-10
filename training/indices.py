import numpy as np
import json

class IIndex:
	def getValue(self, bands):
		pass

def getIndexName(index: IIndex, arg_mapping: dict) -> str:
	name = index.__class__.__name__ + "("
	for arg in index.args:
		name += f"{arg_mapping[arg]}, "
	name = name[:-2] + ")"

	return name

class B(IIndex):
	@staticmethod
	def argsCount():
		return 1;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		return bands[self.args[0]]
	
class NORMP(IIndex):
	@staticmethod
	def argsCount():
		return 2;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		return (a - b) / (a + b)
	
class DIST2(IIndex):
	@staticmethod
	def argsCount():
		return 2;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		return np.sqrt(a*a + b*b)
	
class NORMP4(IIndex):
	@staticmethod
	def argsCount():
		return 4;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		d = bands[self.args[3]]
		return (a - b) / (c + d)
	
class NORPP4(IIndex):
	@staticmethod
	def argsCount():
		return 4;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		d = bands[self.args[3]]
		return (a + b) / (c + d)

class FRAC3(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return (a - c) / (1 + b - c)
	
class FRAC4(IIndex):
	@staticmethod
	def argsCount():
		return 4;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		d = bands[self.args[3]]
		return (a - c) / (1 + b - d)
	
class NORMP3(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return (a - b) / (2 + a +  b - 2 * c)
	
class CVIbased(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return a * b / (c * c)

class NMDIbased(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return (a - b + c) / (a + b + c)
	
class MCARIbased(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return ((a - b) - 0.2 * (a - c)) * a / b

class Hue(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return np.arctan((2 * a - b - c) * (b - c) / 30.5)

class HueSimp(IIndex):
	@staticmethod
	def argsCount():
		return 3;

	def __init__(self, args):
		self.args = args;

	def getValue(self, bands):
		a = bands[self.args[0]]
		b = bands[self.args[1]]
		c = bands[self.args[2]]
		return (2 * a - b - c) * (b - c)

class IndicesClassEncoder:
	def __init__(self, indices_classes, bands_list):
		self.classes = indices_classes
		self.bands = bands_list
		self.band_count = len(self.bands)
		self.total_length = 0
		for cls in self.classes:
			self.total_length += int(self.band_count**cls.argsCount())

	def getIndex(self, index_id: int) -> IIndex:
		index_id = index_id % self.total_length

		for cls in self.classes:
			args_count = cls.argsCount()
			class_length = self.band_count**args_count
			if (index_id >= class_length):
				index_id -= class_length
				continue

			args = [0]*args_count
			for i in range(args_count):
				args[i] = self.bands[int(index_id % self.band_count)]
				index_id = index_id // self.band_count
			return cls(args)

class IndicesClassEncoderEq:
	def __init__(self, indices_classes, bands_list):
		self.classes = indices_classes
		self.bands = bands_list
		self.band_count = len(self.bands)

		max_args_per_class = 0
		for cls in self.classes:
			if cls.argsCount() > max_args_per_class:
				max_args_per_class = cls.argsCount()

		self.length_per_class = (self.band_count ** max_args_per_class)
		self.total_length = len(self.classes) * self.length_per_class

	def getIndex(self, index_id: int) -> IIndex:
		index_id = index_id % self.total_length
		class_id = index_id // self.length_per_class
		index_id = index_id - class_id * self.length_per_class

		cls = self.classes[class_id]
		args_count = cls.argsCount()
		class_length = self.band_count**args_count
		index_id = index_id % class_length

		args = [0]*args_count
		for i in range(args_count):
			args[i] = self.bands[int(index_id % self.band_count)]
			index_id = index_id // self.band_count
		
		return cls(args)


class BakedIndiceClassEncoder:
	def __init__(self, indices_classes, bands_list):
		self.classes = indices_classes
		self.bands = bands_list
		self.band_count = len(self.bands)
		self.private_total_length = 0
		for cls in self.classes:
			self.private_total_length += int(self.band_count**cls.argsCount())

		self.total_length = self.private_total_length

	def getIndex(self, index_id: int) -> IIndex:
		index_id = index_id % self.private_total_length
		return self.instances[index_id][0]

	def private_get_index(self, index_id):
		index_id = index_id % self.private_total_length

		for cls in self.classes:
			args_count = cls.argsCount()
			class_length = self.band_count**args_count
			if (index_id >= class_length):
				index_id -= class_length
				continue

			args = [0]*args_count
			for i in range(args_count):
				args[i] = self.bands[int(index_id % self.band_count)]
				index_id = index_id // self.band_count
			return cls(args)

	def bake(self, informativeness_func, data_class_0, data_class_1):
		self.instances = [None]*self.private_total_length
		self.inform_dict = {}
		for i in range(self.private_total_length):
			feature = self.private_get_index(i)
			value_class_0 = feature.getValue(data_class_0)
			value_class_1 = feature.getValue(data_class_1)

			inform = informativeness_func(value_class_0, value_class_1)
			self.instances[i] = (feature, inform)

		self.instances = sorted(self.instances, key = lambda x: x[1], reverse=True)
		for i in range(self.private_total_length):
			self.inform_dict[i] = self.instances[i][1]
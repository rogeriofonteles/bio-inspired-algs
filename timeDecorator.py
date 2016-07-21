import timeit

class TimeDecorator():
	
	@staticmethod
	def wrapper(funct, *args, **kwargs):
		def wrapped():
			return funct(*args, **kwargs)
		return wrapped

	@classmethod
	def time(cls, funct, *args, **kwargs):
		wrapped = cls.wrapper(funct, *args, **kwargs)
		return timeit.timeit(wrapped, number=1)
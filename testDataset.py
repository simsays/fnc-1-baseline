import sys
import numpy as np
from utils.dataset import DataSet


def main():
	dataset = DataSet()
	#print dataset.articles.keys()[132]
	#print dataset.articles[dataset.articles.keys()[1]]
	print
	print
	print
	print
	print dataset.stances[5]
	print type(dataset.stances[5])
	print dataset.articles[154]

if __name__ == '__main__':
    main()
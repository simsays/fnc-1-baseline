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
	for i in range(10,20):
		print dataset.stances[i]

if __name__ == '__main__':
    main()
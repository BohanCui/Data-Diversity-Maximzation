
#for creating gaussian distribution datasets
import numpy as np
import random
from .group import *
from .utils import *
from .stat_tracker import *
import argparse
from sklearn.datasets import make_blobs


class GenerateDS:
    def __init__(self, num_groups, stat_tracker):
        """
        @params
            num_groups: number of groups that data points may belong to
            stat_tracker: instance of stat_tracker
        """
        self.data_points = []
        self.num_groups = num_groups
        self.stat_tracker = stat_tracker
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, position):
        return self.data_sources[position]
    
    def __str__(self):
        s =  ", Length: " + str(len(self))
        s += ", Points: " + str(self.data_points) + ", Stats: "
        return s + str(self.stat_tracker) + "}"
    
    def __repr__(self):
        return (str(self))
    
    def probability(self, subgroup):
        return self.stat_tracker[subgroup] / len(self)
    
    def add_point(self, data_point):
        self.data_points.append(data_point)
        if self.stat_tracker is not None:
            self.stat_tracker.add_point(data_point)
    
    def sample(self):
        """
        @returns a random data point sampled uniformly
        """
        return np.random.choice(self.data_points)
    


def generate_blobs(size: int, group: int, dim: int) -> None:
    X, y = make_blobs(n_samples=size, centers=10, n_features=dim, random_state=17)
    g = np.random.randint(group, size=size, dtype=np.uint8)
    with open("datasets/blobs_n" + str(size) + "_m" + str(group) + ".csv", "a") as fileobj:
        for i in range(size):
            fileobj.write(str(i) + ",")
            fileobj.write(str(g[i]) + ",")
            for j in range(dim - 1):
                fileobj.write(format(X[i][j], ".6f") + ",")
            fileobj.write(format(X[i][dim - 1], ".6f") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size", default=1000, type=int)
    parser.add_argument("-m", "--group", default=2, type=int)
    parser.add_argument("-d", "--dim", default=2, type=int)
    args = parser.parse_args()
    generate_blobs(args.size, args.group, args.dim)

    #Generate data sets following Gausssian distribution
    def generateDataSets(mean: int, dataSize:int):
        #loc- mean, default as 0
        #scale- sd, default as 1
        data = np.random.normal(loc=mean, scale=1, size=dataSize)


    def testDS:
        #generate sample of 200 values that follow a normal distribution with different mean
        #mean of 0
        dataOrg1= generateDataSets(0,200)
        dataOrg2= generateDataSets(0,200)
        dataOrg3= generateDataSets(0,200)
        dataOrg4= generateDataSets(0,200)
        dataOrg5= generateDataSets(0,200)
        #mean of 1
        data11= generateDataSets(1,200)
        data12= generateDataSets(1,200)
        data13= generateDataSets(1,200)
        data14= generateDataSets(1,200)
        data15= generateDataSets(1,200)
        #mean of 2
        data21= generateDataSets(2,200)
        data22= generateDataSets(2,200)
        data23= generateDataSets(2,200)
        data24= generateDataSets(2,200)
        data25= generateDataSets(2,200)


        #test the results of algorithm1 with epsilon fix with 0.1, see how the mean changes changes the performace


    def  testAlg1:
        eps1 = StreamDivMax(dataOrg1,100,Callable[[1,1],0.1],0.1,0.9,0.1)
        




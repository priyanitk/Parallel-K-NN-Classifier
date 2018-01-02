# Parallel-K-NN-Classifier
 k-Nearest Neighbor (k-NN) is a classification algorithm used in machine learning and data mining applications such as email spam filtering, content retrieval , customer segmentation
in online shopping websites etc.
High parallelism can be achieved using GPU and in comparatively lesser cost than CPU.
So we are trying to do the same task in GPU which speedup the execution by 99%.<p>We implement the parallel program of K-NN classifier .We
write the two kernel module one for calculating the Euclidian distance and other for findind k nearest neighbour .It have mainly three module which are:</p>
<li>Distance calculation</li>
<li>Nearest Neighbour</li>
<li>Class Prediction</li>
<br><p>We have considered the SHUTTLE numeric dataset from
the website of Machine Learning Repository. The dataset
consists of 43,500 training data and 14,500 testing data, each
having nine attributes without any missing components.<p>We found a significant speedup of
the algorithm specially when the test data size
becomes larger. For a test data set of 14,000, we
could observe around 70x speedup in GPU.

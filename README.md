# obm-nn

This project is built as a part of CS6410 course at Northeastern University. In this project, we attempt to train a Feed-Forward Neural Network to solve the Online Bipartite Matching (OLBM) problem after [Alomrani _et al_ 2022](https://arxiv.org/pdf/2109.10380.pdf#appendix.D). We compare this approach to a traditional Greedy Algorithm.

Both the FFNetwork and the Greedy agent are specified in the Agents subdirectory. The dataset we use to generate OLBM instances is derived from the [gMision project](https://gmission.github.io/) - we obtained a pre-processed version of this dataset from [Elias Khalil](https://github.com/lyeskhalil/CORL/tree/master/data). Data, and custom classes to load the data and generate OLBM Instances can be found in the Data subdirectory.

Running the project should be very easy - just pull the code and run train_models.py after installing the required dependencies. This will train three FFNets with different "reward modes" - basically, different ways we tested providing rewards to the RL Agent for different sized OLBM problems. Then, run test_models.py to analyze the performance of each trained network vs. a greedy agent.

This project was implemented primarily with [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/) by David Anderson, Bernhard Sander, Akhila Sulgante and Kasi Viswanath Vandanapu.

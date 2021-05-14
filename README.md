# Drought watch prediction
Weights and Biases's Public Benchmark project - Drought Watch for FSDL Spring 2021 course final project

1. Drought_Prediction_Iteration_1.ipynb -This is the notebook used for training the landsat dataset using multiple architectures. Crux is taken from pytorch's transfer learning tutorial. I changed values in config to train using each model architecture. Other than that, name of weights file that gets saved should be changed per each architecture.
2. Random_Sweeps_on_Drought_Watch_Densenet.ipynb - Similar to above when it comes to trying with different architecture. W&B's hyper parameter sweeps is integrated in this notebook. Search method used is "random".
3. Bayesian_Sweeps_on_Drought_Watch_Densenet.ipynb - Similar to above when it comes to trying with different architecture. W&B's hyper parameter sweeps is integrated in this notebook. Search method used is "bayes".

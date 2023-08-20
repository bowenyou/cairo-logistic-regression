# Verifiable Logistic Regression 

This is a submission for the ETHToronto hackathon (2023).

Please see the `notebooks` directory for the Python code of synthetic dataset generation as well as generation of dataset in Cairo.
The process of solving the logistic regression problem is present as well.

The `src` directory contains the Cairo code. We make use of the Orion framework for tensor manipulation.
The `train.cairo` file contains the functions to run gradient descent in Cairo. This can be used as a scaffold for verifiable deep learning.

We use `Scarb` for managing the installation of Cairo.

To run the tests:
```
scarb test
```

The output of the test should correspond to the same result in the Jupyter notebooks. 
I set the number of iterations to be `100` to keep the script consise. Running it for more iterations would lead to a further decrease in loss and better results.
Learning rate `alpha` and number of iterations are parameters that can be specified by the user.

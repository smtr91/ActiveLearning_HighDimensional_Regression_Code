import numpy as np
import os
import pandas as pd
from scipy.interpolate import *
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.space import Space

from model import SPGD
from utils import *

## Main Function
def active_learning():
    """
    This function performs a run of an active learning loop for surrogate modeling
    using a SPGD model and the high dimensional active learning criterion.

    Steps:
    1. Create a D-dimensional uniform grid of points ("pool") to serve as candidate data points.
    2. Generate an initial training dataset using a space-filling design (Halton or Hammersly sequence) and 
        train an initial surrogate model (`learner_model`).
    4. Generate a separate dataset (`df_criterion_train`) to train a model of the criterion (`criterion_model`).
    5. Iteratively select new points to add to the training set using the criterion model and update both models. 
    6. Evaluate the performance of the surrogate model (R² score between predicted and true values on a test set).

    """
    # Configuration parameters for SPGD model
    spgd_params = {}
    spgd_params['nFun'] = int(input("Enter number of basis functions (e.g., 6): "))                 # Number of basis functions
    spgd_params['nModes'] = int(input("Enter number of modes (e.g., 1): "))                         # Number of modes
    spgd_params['activeDim'] = int(input("Enter number of active dimensions (e.g., 5): "))         # Number of active dimensions

    # Experiment setup - asking user for values
    initial_sample_size = int(input("Enter number of initial training samples (e.g., 25): "))       # Number of initial training samples
    num_queries = int(input("Enter number of active learning steps (e.g., 20): "))                   # Number of active learning steps
    criterion_sample_size = int(input("Enter number of samples to train criterion sub-model (e.g., 5000): "))  # Criterion sub-model training size
    input_dim = int(input("Enter dimensionality of the input space (e.g., 5): "))                  # Dimensionality of the input space
    grid_subdivision = int(input("Enter number of grid subdivisions per dimension (e.g., 15): "))    # Grid resolution for pool generation
    
    # Selection strategy input (validated)
    while True:
        selection_strategy = input("Enter selection strategy ('one', 'cross' or 'simplex'): ").strip().lower()  # Strategy to select query points
        if selection_strategy in ['one', 'cross', 'simplex']:
            break
        else:
            print("Invalid selection. Please enter 'one', 'cross' or 'simplex'.")

    # Define the pool: D-dimensional grid of points
    df_pool_grid = pd.DataFrame(itertools.product(np.linspace(-0.5, 0.5, grid_subdivision), repeat=input_dim), columns=[f'x_{i}' for i in range(input_dim)])
    df_pool_grid['y'] = output(df_pool_grid.filter(regex='^x_').values)  # Evaluate the function
    df_pool_grid['flag'] = 'pool'
    df_pool_grid['step'] = 0
    spgd_params['Ranges'] = df_pool_grid.filter(regex='^x_').describe().loc[['min', 'max']].values
    print(f'grid size = {df_pool_grid.shape[0]}')

    # Initialization of the training set using Halton/Hammersly sequence
    sampling_method = input("Choose initialization method ('halton' or 'hammersly'): ").strip().lower()
    space = Space([(-0.5, 0.5)] * input_dim) 
    if sampling_method == 'hammersly':
        sampler = Hammersly()
    elif sampling_method == 'halton':
        sampler = Halton()
    else:
        print("Invalid method selected. Defaulting to Halton sequence.")
        sampler = Halton()
    initial_design = sampler.generate(space.dimensions, initial_sample_size)
    # Convert to DataFrame and evaluate
    df_train_init = pd.DataFrame(initial_design, columns=[f'x_{i}' for i in range(input_dim)])
    df_train_init['y'] = output(df_train_init.filter(regex='^x_').values)
    df_train_init['flag'] = 'train'
    df_train_init['step'] = initial_sample_size

    # Combine initial training data and pool into a unified dataset
    df_all = pd.concat([df_pool_grid, df_train_init], ignore_index=True)

    # Train initial surrogate model
    learner_model = SPGD(spgd_params['Ranges'], df_train_init.filter(regex='^x_').values, df_train_init.y.values, spgd_params['nFun'], spgd_params['nModes'], spgd_params['activeDim'])

    # Define test set (reusing pool grid)
    df_test = df_all
    df_test['y'] = output(df_test.filter(regex='^x_').values)

    # Build initial training data for criterion learner using LHS
    df_criterion_train = lhs_criterion(learner_model, spgd_params['Ranges'], criterion_sample_size)
    # Initialize criterion model
    criterion_model = SPGD(spgd_params['Ranges'], df_criterion_train.filter(regex='^x_').values, df_criterion_train.y.values, spgd_params['nFun'], spgd_params['nModes'], spgd_params['activeDim'])

    R2_scores = {}
    query_steps = [i for i in range(initial_sample_size + 1, num_queries + initial_sample_size + 1)]
    print(f'start run', flush=True)

    # Creation of a file to store the results (here R² but can be replace with anything else you want, you can also extract the updated model or training database for example)
    file_out = open(f'results.csv', 'w') 
    for step in query_steps:
        print(f'{step=}', flush=True)
        # Query and add new points to training set
        df_all, learner_model = update_functioncriterion(df_all, criterion_model, spgd_params, step, input_dim, selection_strategy)
        # Update criterion values using new surrogate model predictions
        df_criterion_train['y'] = criterion(learner_model, df_criterion_train.filter(regex='^x_'))[1]
        # Retrain criterion model
        criterion_model = SPGD(spgd_params['Ranges'], df_criterion_train.filter(regex='^x_').values, df_criterion_train.y.values, spgd_params['nFun'], spgd_params['nModes'], spgd_params['activeDim'])
        # Predict on test set using updated surrogate model
        df_test[f'y_pred_{step}'] = learner_model.predict(df_test.filter(regex='^x_').values)
        # Calculate R² correlation between predictions and ground truth on pool
        R2_scores = correlation( df_test.loc[df_test.flag == 'pool', [f'y_pred_{step}']].values, df_test.loc[df_test.flag == 'pool', ['y']].values)
        file_out.write(f'{step}, {R2_scores}\n')
        file_out.flush()
    file_out.close()
    # return learner_model (or other)

def main():
    print(f'==> Running code in {os.getcwd()}', flush=True) 
    active_learning()

if __name__ == "__main__":
    main()
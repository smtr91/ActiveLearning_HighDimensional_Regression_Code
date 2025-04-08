import numpy as np
import itertools
import pyDOE2
from model import SPGD
import pandas as pd

## Function Studied
def output(input_array):
    """
    Computes a custom output function for each row (input vector) in the input_array.
    This is filled as example with a 5D polynomial function.
    Parameters:
        input_array (list or np.ndarray): A 2D array where each row is an input vector of at least input_dim features.
    Returns:
        list: A list of output values, one for each input vector.
    Notes:
        For general use this function should be replaced or adapted to any other desired output function (batch launch of simulation).
    """
    output_values = []
    for row in input_array:
        result = ((8 * row[0]**3 - 6 * row[0] - 0.5 * row[1])**2 + (4 * row[2]**3 - 3 * row[2] - 0.25 * row[3])**2 + 0.1 * (row[4]**2 - 1)) #replace here by your output
        #result = os.system('"C:/Users/Programmes/VPSolution-2022.01_Solvers_Windows-Intel64/Solver/bin/pamcrash.bat" -np 4 -fp 1 beam.pc')
        output_values.append(result)
    return output_values

## Active Learning High Dimensional Criterion Definition
def criterion(learner, X_pool):
    """
    Computes our active learning criterion for each point in the pool X_pool
    for a given SPGD learner (surrogate model), and returns:
    - The index of the point with the highest criterion value.
    - A list of all criterion values for the pool.   
    This criterion measures how much a new sample would contribute to
    improving the surrogate model based on its functional basis.
    Args:
        learner (SPGD): The current surrogate model.
        X_pool (DataFrame): Pool of candidate points (samples) to evaluate.
    Returns:
        idx_max (int): Index of the point in X_pool with maximum criterion.
        criterion (list): List of scalar criterion values for each point.
    """
    # Pre-allocate basis evaluations
    N = np.zeros((*X_pool.shape, learner.nFun))    # N: functional basis evaluations (nFun per dimension)
    X = np.zeros((*X_pool.shape, learner.nModes))  # X: modal basis evaluations (nModes per dimension)
    # Compute functional and modal basis for each point and dimension
    for dim in range(learner.nDim):
        N[:, dim, :] = learner.N[dim](X_pool.iloc[:, dim])
        X[:, dim, :] = learner.X[dim](X_pool.iloc[:, dim])
    # J matrix: represents the tensor product of basis functions
    J = np.ones((learner.nDim * learner.nModes * learner.nFun, X_pool.shape[0]))
    for i, (dim, mode, func) in enumerate(itertools.product(range(learner.nDim), range(learner.nModes), range(learner.nFun))):
        for ax in range(learner.nDim):
            if dim == ax:
                J[i] *= N[:, ax, func]
            else:
                J[i] *= X[:, ax, mode]
    # Compute criterion for each sample: sum of squared tensor product basis
    criterion = []
    for k in range(J.shape[1]):
        criterion.append(J[:, k].T @ J[:, k])
    # Get index of the point with maximum criterion value (most "informative")
    idx_max = X_pool.iloc[np.argmax(criterion)].name
    return idx_max, criterion

## Group functions
# Group Definitions
def simplex(center_coordinates, distance):  
    """
    Generate a group of points forming a simplex around a center point.
    Args:
        center_coordinates (array-like): Coordinates of the center point.
        distance (float): Distance from the center to each vertex of the simplex.
    Returns:
        np.ndarray: Array of shape (n+1, n) representing the n+1 vertices of the simplex
                    in an n-dimensional space, including the center.
    """
    dim = len(center_coordinates)
    num_vertices = dim + 1
    group = np.zeros((num_vertices, dim))
    group[0] = center_coordinates
    for i, value in enumerate(center_coordinates):
        group[i + 1][i] = value + distance
        direction = group[i + 1] - center_coordinates
        group[i + 1] = center_coordinates + (direction / np.linalg.norm(direction)) * distance
    return group  # List of coordinates forming the simplex

def cross(center_coordinates, distance):
    """
    Generate a group of points forming a cross pattern around a center point.
    Args:
        center_coordinates (array-like): Coordinates of the center point.
        distance (float): Distance to move along each dimension (positive and negative).
    Returns:
        list: List of points including the center and 2 * dim neighbors.
    """
    dim = len(center_coordinates)
    group = []
    for i in range(dim):
        new_point_plus = center_coordinates.copy()
        new_point_plus[i] += distance
        group.append(new_point_plus)

        new_point_minus = center_coordinates.copy()
        new_point_minus[i] -= distance
        group.append(new_point_minus)
    group.append(center_coordinates)
    return group  # List of coordinates forming the Cross group

# Criterion value of a group
def criterion_one_group_cross(point, learner_criterion, distance):
    """
    Evaluate the sum of criterion values over a cross group around a given point, giving the corresponding group criterion value.
    Args:
        point (Series): Pool point around which to form the cross group.
        learner_criterion (SPGD): Surrogate model used to compute the criterion.
        distance (float): Distance to define the neighborhood.
    Returns:
        float: Sum of predicted criterion values over the group.
    """
    group_points = cross(point, distance)
    total_criterion = 0
    # Sum predicted values from criterion model over the cross group
    for neighbor in group_points:
        prediction = learner_criterion.predict(np.array([neighbor.filter(regex='^x_').values.tolist()]))[0][0]
        total_criterion += prediction
    return total_criterion

def criterion_one_group_simp(point, learner_criterion, distance):
    """
    Evaluate the sum of criterion values over a simplex group around a given point, giving the corresponding group criterion value.
    Args:
        point (Series): Pool point around which to form the simplex group.
        learner_criterion (SPGD): Surrogate model used to compute the criterion.
        distance (float): Distance to define the neighborhood.
    Returns:
        float: Sum of predicted criterion values over the group.
    """
    group_points = simplex(point, distance)
    total_criterion = 0
    # Sum predicted values from criterion model over the simplex group
    for neighbor in group_points:
        prediction = learner_criterion.predict(np.array([neighbor.tolist()]))[0][0]
        total_criterion += prediction
    return total_criterion

## Update Step by Step
def criterion_cross(df, d, learner):
    """
    Select the best cross group based on the group criterion.
    Args:
        df (DataFrame): Global dataset containing training and pool points.
        d (float): Distance used to generate the cross group.
        learner (SPGD): Learner model used to evaluate the criterion.
    Returns:
        tuple: List of group points and the updated dataset with selected center marked as train.
    """
    df_pool = df[df.flag == 'pool'].filter(regex='^x_')
    df_std = df_pool.swifter.apply(lambda x: criterion_one_group_cross(x, learner, d), axis=1)
    idx_max = np.argmax(df_std.values)
    point = df.loc[idx_max].filter(regex='^x_').values
    x_group = cross(point, d)
    df.loc[idx_max, 'flag'] = 'train'
    df.loc[idx_max, 'step'] = df[df.flag == 'train'].shape[0]
    return x_group, df

def criterion_simp(df, d, learner):
    """
    Select the best simplex group based on the group criterion.
    Args:
        df (DataFrame): Global dataset containing training and pool points.
        d (float): Distance used to generate the simplex group.
        learner (SPGD): Learner model used to evaluate the criterion.
    Returns:
        tuple: List of group points and the updated dataset with selected center marked as train.
    """
    df_pool = df[df.flag == 'pool'].filter(regex='^x_')
    df_std = df_pool.swifter.apply(lambda x: criterion_one_group_simp(x, learner, d), axis=1)
    idx_max = np.argmax(df_std.values)
    point = df.loc[idx_max].filter(regex='^x_').values
    x_group = simplex(point, d)
    df.loc[idx_max, 'flag'] = 'train'
    df.loc[idx_max, 'step'] = df[df.flag == 'train'].shape[0]
    return x_group, df

def update_functioncriterion(df, learner_std, spgd_params, q, D, group):
    """
    Adds new training points to the dataset based on the high dimensional active learning strategy.
    Depending on the specified 'group' type ('one', 'cross', or 'simplex'), this function:
        - Selects the next best point or group of points based on the criterion.
        - Updates the dataset with new training points.
        - Re-trains the SPGD surrogate model using the augmented training set.
    Args:
        df (DataFrame): Current dataset including training and pool points.
        learner_std (SPGD): Surrogate model used to compute the acquisition criterion.
        spgd_params (dict): Dictionary of parameters for the SPGD learner.
        q (int): Current active learning iteration index (used for step labeling).
        D (int): Dimensionality of the input space.
        group (str): Acquisition strategy: 
                     'one'     -> add a single point with the best criterion,
                     'cross'   -> add a group of (2*D + 1) points in a cross shape,
                     'simplex' -> add a group of (D + 1) points forming a simplex.
    Returns:
        tuple: Updated dataset (df) and updated surrogate model (learner).
    """
    if group == 'one':  # Select and add a single point with the highest criterion
        idx_max = criterion(learner_std, df[df.flag == 'pool'].filter(regex='^x_'))[0]
        df.loc[idx_max, 'flag'] = 'train'
        df.loc[idx_max, 'step'] = df[df.flag == 'train'].shape[0]
    elif group == 'cross':  # Add a cross-shaped group: (2*D + 1) points
        list_solution, df = criterion_cross(df, 0.1, learner_std)
        for k in range(len(list_solution)):
            new_row = {f'x_{i}': list_solution[k][i] for i in range(D)}
            new_row['y'] = output([list_solution[k]])[0]  # Evaluate true output
            new_row['flag'] = 'train'
            new_row['step'] = q
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    elif group == 'simplex':  # Add a simplex group: (D + 1) points
        list_solution, df = criterion_simp(df, 0.1, learner_std)
        for k in range(len(list_solution)):
            new_row = {f'x_{i}': list_solution[k][i] for i in range(D)}
            new_row['y'] = output([list_solution[k]])[0]  # Evaluate true output
            new_row['flag'] = 'train'
            new_row['step'] = q
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Retrain the SPGD model on the updated training data
    learner = SPGD(spgd_params['Ranges'], df[df.flag == 'train'].filter(regex='^x_').values, df[df.flag == 'train'].y.values, spgd_params['nFun'],spgd_params['nModes'],spgd_params['activeDim'])
    return df, learner

## Hypercube Latin Sampling
def lhs_criterion(learner, ranges, samples):
    """
    Generates a Latin Hypercube Sampling (LHS) design and evaluates the hight dimensional criterion value for each sample.
    Args:
        learner (SPGD): Learner model used to compute criterion.
        ranges (ndarray): Array of shape (2, D) with min and max bounds for each dimension.
        samples (int): Number of LHS samples to generate.
    Returns:
        DataFrame: Dataset containing LHS points with computed criterion values.
    """
    dim = ranges.shape[1]  # number of dimensions
    df = pd.DataFrame(pyDOE2.lhs(dim, samples), columns=[f'x_{i}' for i in range(dim)])
    df *= ranges[1] - ranges[0]  # scale to the problem range
    df += ranges[0]  # shift to the correct range
    df['y'] = criterion(learner, df.filter(regex='^x_'))[1]  # evaluate criterion
    df['flag'] = 'train'
    df['step'] = samples
    return df

## Evaluation metrics 
# Correlation
def correlation(variable1, variable2):
    """
    Computes the Pearson correlation coefficient between two variables.
    Args:
        variable1 (array-like): First variable.
        variable2 (array-like): Second variable.
    Returns:
        float: Pearson correlation coefficient between variable1 and variable2.
    """
    covariance_numerator = 0      # Covariance of variable1 and variable2
    variance_variable1 = 0        # Variance of variable1
    variance_variable2 = 0        # Variance of variable2
    N = len(variable1)
    mean_variable1 = np.mean(variable1)
    mean_variable2 = np.mean(variable2)
    for k in range(N):
        covariance_numerator += (variable1[k] - mean_variable1) * (variable2[k] - mean_variable2)
        variance_variable1 += (variable1[k] - mean_variable1) ** 2
        variance_variable2 += (variable2[k] - mean_variable2) ** 2
    denominator = np.sqrt(variance_variable1 * variance_variable2)  # Product of standard deviations
    return (covariance_numerator / denominator)[0]      # Return scalar correlation value

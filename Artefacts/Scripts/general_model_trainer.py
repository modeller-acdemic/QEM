# -*- coding: utf-8 -*-
"""
This code simply trains the random forest regressor using the feature-engineered data from the collector
It also verifies the accuracy of the feature engineering by computing the RMSE from the raw data, too
- it uses a different test pauli observable deriviation to ensure that the result was not caused by an error in the calculation.
Finally, it saves the model to file for results analysis.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor # for robustness test
from sklearn.ensemble import GradientBoostingRegressor # for robustness test
from sklearn.tree import DecisionTreeRegressor # for robustness test
from sklearn.gaussian_process import GaussianProcessRegressor # for robustness test
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel # for robustness test
from sklearn.preprocessing import StandardScaler # For the MLP
from sklearn.metrics import mean_squared_error
import json
from qiskit.quantum_info import DensityMatrix, SparsePauliOp
from qiskit.result import Counts
import joblib

# hyperparameter tuning
from skopt.sampler import Lhs
from mitigation_script import main as run_test 

# compute per circuit error from the already transformed expectation values
def transformed_error(results_df):
    transformed_errors = []
    
    # iterate over groups of rows that share a circuit index
    for _, group in results_df.groupby('circuit_index'):
        
        # extract the vector of ideal expectation values
        ideal_vector = group['exp_val_ideal'].values
        
        # extract the vector of noisy expectation values
        noisy_vector = group['exp_val_noisy'].values
        
        # compute the RMSE between the vectors
        rmse = np.sqrt(np.mean((ideal_vector - noisy_vector) ** 2))
        
        # append the RMSE as a float to the output list
        transformed_errors.append(float(rmse))
    
    return transformed_errors

# compute the average per circuit mitigation error
def mitigation_error(results_dataframe, feature_matrix, fitted_model):
    per_circuit_mitigation_error_errors = []
    
    # create a copy of the results dataframe to avoid mutating the caller input
    results_with_predictions = results_dataframe.copy()
    
    # generate model predictions for every row in the feature matrix
    predicted_expectation_values_full = fitted_model.predict(feature_matrix)
    
    # attach the predictions to the copied dataframe
    results_with_predictions['predicted_exp_val'] = predicted_expectation_values_full
    
    # iterate over groups of rows that share a circuit index
    for circuit_index, circuit_group in results_with_predictions.groupby('circuit_index'):
        
        # extract the ideal and predicted expectation values
        ideal_vector_per_circuit = circuit_group['exp_val_ideal'].values
        predicted_vector_per_circuit = circuit_group['predicted_exp_val'].values
        
        # compute the RMSE between the ideal and predicted vectors
        rmse_per_circuit = np.sqrt(np.mean((ideal_vector_per_circuit - predicted_vector_per_circuit) ** 2))
        per_circuit_mitigation_error_errors.append(float(rmse_per_circuit))

    # compute the average per circuit mitigation error
    average_mitigation_error = float(np.mean(np.asarray(per_circuit_mitigation_error_errors, dtype=float)))
    
    return average_mitigation_error

# compute per circuit errors using qiskit density matrices as the ground truth
def raw_error(raw_df):       
    raw_circuit_errors = []
    
    # iterate over each row sorted by circuit index for deterministic ordering
    for _, row in raw_df.sort_values('circuit_index').iterrows():
        qiskit_ideal_vector = []
        qiskit_noisy_vector = [] 
        
        # load the noisy count dictionary from its json string
        noisy_counts_dict = json.loads(row['noisy_counts'])
        
        # load the ideal count dictionary from its json string
        ideal_counts_dict = json.loads(row['ideal_counts'])
        
        # create qiskit counts object for noisy counts
        noisy_counts = Counts(noisy_counts_dict)
        
        # create qiskit counts object for ideal counts
        ideal_counts = Counts(ideal_counts_dict)
        
        # choose the number of qubits based on the keys present
        if ideal_counts:
            num_qubits = len(next(iter(ideal_counts)))
        elif noisy_counts:
            num_qubits = len(next(iter(noisy_counts)))
        else:
            num_qubits = 0
        
        # build the probability dictionary for ideal counts
        ideal_probs = {}
        for state, count in ideal_counts.items():
            ideal_probs[state] = count / ideal_counts.shots()
        
        # build the probability dictionary for noisy counts
        noisy_probs = {}
        for state, count in noisy_counts.items():
            noisy_probs[state] = count / noisy_counts.shots()
        
        # build the diagonal entries for the ideal density matrix
        ideal_diag = [ideal_probs.get(f"{i:0{num_qubits}b}", 0.0) for i in range(2**num_qubits)]
        
        # build the diagonal entries for the noisy density matrix
        noisy_diag = [noisy_probs.get(f"{i:0{num_qubits}b}", 0.0) for i in range(2**num_qubits)]
        
        # create the ideal density matrix
        ideal_state = DensityMatrix(np.diag(ideal_diag))
        
        # create the noisy density matrix
        noisy_state = DensityMatrix(np.diag(noisy_diag))
        
        # iterate through each single qubit pauli z observable
        for i in range(num_qubits):
            
            # build the pauli string for this observable
            pauli_string = ['I'] * num_qubits
            pauli_string[num_qubits - 1 - i] = 'Z'
            
            # create the sparse pauli operator
            op = SparsePauliOp("".join(pauli_string))
            
            # append the real part of the expectation value of the operator under the ideal state
            qiskit_ideal_vector.append(float(np.real(ideal_state.expectation_value(op))))
    
            # append the real part of the expectation value of the operator under the noisy state
            qiskit_noisy_vector.append(float(np.real(noisy_state.expectation_value(op))))
            
        # compute the RMSE between the ideal and noisy vectors
        rmse = np.sqrt(np.mean((np.array(qiskit_ideal_vector) - np.array(qiskit_noisy_vector)) ** 2))
 
        # append the distance to the output list
        raw_circuit_errors.append(float(rmse))
    
    return raw_circuit_errors

# compute percentage agreement using numpy isclose on two sequences
def compute_agreement_percentage(transformed_errors, raw_errors):
    
    # convert the transformed errors to a numpy array of floats
    transformed_array = np.asarray(transformed_errors, dtype=float)
    
    # convert the raw errors to a numpy array of floats
    raw_array = np.asarray(raw_errors, dtype=float)
    
    # compute elementwise closeness between the two arrays
    closeness_mask = np.isclose(transformed_array, raw_array)
    
    # compute the percentage of agreement across all elements
    agreement_percentage = float(closeness_mask.mean() * 100.0)
    
    return agreement_percentage

# construct features and target from the results dataframe
def features_target(results_df):
    
    # create a defensive copy to avoid mutating the caller dataframe
    df = results_df.copy()
    
    # define non feature columns that must be excluded
    non_feature_cols = ['circuit_index', 'observable', 'exp_val_ideal', 'n', 'd']
    
    # choose base feature columns by excluding non feature columns
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # compute the error target as ideal minus noisy
    # df['error'] = df['exp_val_ideal'] - df['exp_val_noisy']
    
    # perform one hot encoding of the observable column
    observable_dummies = pd.get_dummies(df['observable'], prefix='obs')
    
    # concatenate the dummy columns to the dataframe
    df = pd.concat([df, observable_dummies], axis=1)
    
    # extend the feature column list with the new dummy columns
    feature_cols.extend(list(observable_dummies.columns))
    
    # extract the feature matrix
    X = df[feature_cols]
    
    # extract the target vector
    y = df['exp_val_ideal']
    
    return X, y, feature_cols

# build a latin hypercube over plain candidate lists and return raw sample vectors
def latin_hypercube(search_description, random_state=42):
    
    # ensure each dimension is a list of candidate values
    dimensions = [list(d) for d in search_description]
    
    # create the latin hypercube sampler
    sampler = Lhs(criterion='maximin', iterations=1000)
    
    # compute the number of trials as the sum of candidate counts across all dimensions
    n_trials = sum(len(d) for d in dimensions)
    
    # generate latin hypercube samples over the provided candidate lists
    samples = sampler.generate(dimensions, int(n_trials), random_state=int(random_state))
    
    return samples

# perform latin hypercube hyperparameter tuning and return the best fitted model
def hyperparameter_tuning(results_df, X, y, fraction, random_state=42, param_dicts=None, estimator_class=None):
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    # iterate over each sampled hyperparameter configuration with a running index
    for design_index, params in enumerate(param_dicts, start=1):
        print(f"latin hypercube trial {design_index}/{len(param_dicts)}")
        print(f"parameters: {params}")
        
        # split the full feature matrix and target vector into training and test partitions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # create the estimator instance using the provided class and the sampled parameters
        model = estimator_class(**params)
        
        # fit the model on the training partition
        model.fit(X_train, y_train)
        
        # build the temporary model filename that the test script expects for this fraction
        temp_model_filename = f"model_{int(fraction*100)}{estimator_class}.joblib"
        
        # save the fitted model so the external test pipeline can load and evaluate it
        joblib.dump(model, temp_model_filename)
        
        # run the external test pipeline and read back the generalised mitigated RMSE
        mean_mitigated_rmse = float(run_test(fraction, model=estimator_class))
        
        # print the observed objective value for this trial
        print(f"generalised mitigated RMSE: {mean_mitigated_rmse}")
        
        # update the incumbent best model when the objective improves
        if mean_mitigated_rmse < best_rmse:
            best_rmse = mean_mitigated_rmse
            best_model = model
            best_params = params
    
    # print the best observed objective value
    print(f"best RMSE {best_rmse:.6f}")
    
    # print the best parameter dictionary
    print(f"best parameters: {best_params}")
    
    return best_model

# perform latin hypercube hyperparameter re-tuning using test set rmse
def hyperparameter_retuning(results_df, X, y, fraction, random_state=42, param_dicts=None, estimator_class=None):
    
    # track the best observed rmse and corresponding model and params
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    # iterate over each sampled hyperparameter configuration with a running index
    for design_index, params in enumerate(param_dicts, start=1):
        print(f"latin hypercube trial {design_index}/{len(param_dicts)}")
        print(f"parameters: {params}")
        
        # split the full feature matrix and target vector into training and test partitions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # create the estimator instance using the provided class and the sampled parameters
        model = estimator_class(**params)
        
        # fit the model on the training partition
        model.fit(X_train, y_train)
        
        # compute predictions for the test partition
        y_test_pred = model.predict(X_test)
        
        # compute the test set root mean squared error
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = float(np.sqrt(test_mse))
        print(f"test set rmse: {test_rmse}")
        
        # update the incumbent best model when the objective improves
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model = model
            best_params = params
    
    # print the best observed objective value
    print(f"best test rmse {best_rmse:.6f}")
    
    # print the best parameter dictionary
    print(f"best parameters: {best_params}")
    
    return best_model

# perform latin hypercube hyperparameter tuning for a random forest model
def rf_tune(results_df, X, y, fraction, random_state=42, retune=False):
    
    # define the candidate lists for each hyperparameter
    dimensions = [
        list(range(100, 1001, 100)),
        list(range(5, 61, 5)),
        list(range(1, 11)),
        list(range(2, 21)),
        ['sqrt', 'log2', 0.3, 0.5, 0.8],
        [True, False]
    ]
    
    # draw sample vectors using the latin hypercube
    sampled_vectors = latin_hypercube(dimensions, random_state=random_state)
    
    # prepare parameter dictionaries
    parameter_dictionaries = []
    for sample in sampled_vectors:
        
        # pick the feature control
        feat_val = sample[4]
        
        # ensure numeric options are float and keep named options as text
        if isinstance(feat_val, str) and feat_val not in {'sqrt', 'log2'}:
            feat_val = float(feat_val)
        
        # assemble the parameter dictionary
        params = {
            'n_estimators': int(sample[0]),
            'criterion': 'squared_error',
            'min_samples_split': int(sample[3]),
            'min_samples_leaf': int(sample[2]),
            'max_depth': int(sample[1]),
            'max_features': feat_val,
            'bootstrap': bool(sample[5]),
            'random_state': int(random_state)
        }

        # collect the parameter dictionary
        parameter_dictionaries.append(params)
    
    # choose the estimator class
    estimator_class = RandomForestRegressor
    
    # run the tuner - check if this is the retuned version or not
    if retune == False:
        best_model = hyperparameter_tuning(results_df, X, y, fraction, random_state=random_state, param_dicts=parameter_dictionaries, estimator_class=estimator_class)
    else:
        best_model = hyperparameter_retuning(results_df, X, y, fraction, random_state=random_state, param_dicts=parameter_dictionaries, estimator_class=estimator_class)
    
    return best_model

# perform latin hypercube hyperparameter tuning for a gradient boosting model
def GBR_tune(results_df, X, y, fraction, random_state=42):
    
    # define the discrete candidate lists for each hyperparameter in a fixed order
    dimensions = [
        list(range(50, 801, 50)),
        [0.01, 0.02, 0.05, 0.1, 0.2],
        list(range(1, 7)),
        list(range(1, 11)),
        list(range(2, 21)),
        [0.6, 0.8, 1.0],
        [None, 'sqrt', 'log2'],
        ['squared_error', 'absolute_error']
    ]
    
    # draw sample vectors using the discrete latin hypercube
    sampled_vectors = latin_hypercube(dimensions, random_state=random_state)
    
    # convert sample vectors into parameter dictionaries
    parameter_dictionaries = []
    for sample in sampled_vectors:
        
        # assemble the gradient boosting parameter dictionary
        params = {
            'n_estimators': int(sample[0]),
            'learning_rate': float(sample[1]),
            'max_depth': int(sample[2]),
            'min_samples_leaf': int(sample[3]),
            'min_samples_split': int(sample[4]),
            'subsample': float(sample[5]),
            'max_features': sample[6],
            'loss': sample[7],
            'random_state': int(random_state)
        }
        
        # append the parameter dictionary to the list
        parameter_dictionaries.append(params)
    
    # select the estimator class for gradient boosting
    estimator_class = GradientBoostingRegressor
    
    # run the generic tuner
    best_model = hyperparameter_tuning(results_df, X, y, fraction, random_state=random_state, param_dicts=parameter_dictionaries, estimator_class=estimator_class)
    
    return best_model

# perform latin hypercube hyperparameter tuning for a decision tree model
def DecisionTree_tune(results_df, X, y, fraction, random_state=42):

    # define the candidate lists for each hyperparameter
    dimensions = [
        ['squared_error', 'friedman_mse', 'absolute_error'],
        ['best', 'random'], 
        list(range(5, 51, 5)) + [None], 
        list(range(2, 21)),
        list(range(1, 21)), 
        ['sqrt', 'log2', None]
    ]

    # draw sample vectors using the discrete latin hypercube
    sampled_vectors = latin_hypercube(dimensions, random_state=random_state)
    
    # convert sample vectors into parameter dictionaries
    parameter_dictionaries = []
    for sample in sampled_vectors:
        params = {
            'criterion': sample[0],
            'splitter': sample[1],
            'max_depth': sample[2],
            'min_samples_split': int(sample[3]),
            'min_samples_leaf': int(sample[4]),
            'max_features': sample[5],
            'random_state': int(random_state)
        }
        parameter_dictionaries.append(params)
    
    # select the estimator class for the decision tree
    estimator_class = DecisionTreeRegressor
    
    # run the generic tuner
    best_model = hyperparameter_tuning(results_df, X, y, fraction, random_state=random_state, param_dicts=parameter_dictionaries, estimator_class=estimator_class)
    
    return best_model

# run the full pipeline from file paths and fraction
def main(fraction=1.0, input_model = ""):
    print(f"Running model training with {int(fraction*100)}% empirical data...\n")
    
    # input data
    results_path = f'../Data/training_data/feature_engineered/engineered_results_{int(fraction*100)}.csv'
    raw_path = f'../Data/training_data/raw_data/raw_outputs_{int(fraction*100)}.csv'
    
    # output model name
    model_filename = f"../Data/models/model_{int(fraction*100)}{input_model}.joblib"
    
    # read the feature engineered results from csv
    results_df = pd.read_csv(results_path)

    # read the raw measurement outputs from csv
    raw_df = pd.read_csv(raw_path)
    
    # compute RMSE from the transformed data
    transformed_circuit_errors = transformed_error(results_df)
    
    # compute and print the average transformed RMSE
    average_RMSE_transformed = float(np.mean(np.asarray(transformed_circuit_errors, dtype=float)))
    print(f"Average RMSE from transformed data: {average_RMSE_transformed}")
    
    # compute RMSE verified through qiskit from raw counts
    raw_circuit_errors = raw_error(raw_df)
    
    # compute and print the average raw RMSE
    average_RMSE_raw = float(np.mean(np.asarray(raw_circuit_errors, dtype=float)))
    print(f"Average RMSE from raw outputs: {average_RMSE_raw}")

    # compute and print the agreement percentage between the two error lists
    agreement_percentage = compute_agreement_percentage(transformed_circuit_errors, raw_circuit_errors)
    print(f"Agreement between transformed and raw error calculations: {agreement_percentage:.2f}%\n")
    
    # build features and target from the results dataframe
    X, y, feature_cols = features_target(results_df)
    
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # hyperparameter tuning - keep toggled off
    # model = rf_tune(results_df, X, y, fraction, random_state=42)
    # model = rf_tune(results_df, X, y, fraction, random_state=42, retune=True)
    # model = GBR_tune(results_df, X, y, fraction, random_state=42)
    # model = DecisionTree_tune(results_df, X, y, fraction, random_state=42)

    # RF
    if input_model == "":
        print("Training RF model")
        model = RandomForestRegressor(
            n_estimators=200,
            criterion='squared_error',
            min_samples_split=10,
            min_samples_leaf=1,
            max_depth=10,
            max_features=0.5,
            bootstrap=True,
            random_state=42)
    
    # GBR
    if input_model == "_GBR":
        print("Training GBR model")
        model = GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.2,
            max_depth=3,
            min_samples_leaf=6,
            min_samples_split=4,
            subsample=0.8,
            max_features='sqrt',
            loss='absolute_error',
            random_state=42
        )
    
    # Decision Tree regressor
    if input_model == "_DT":
        print("Training DT model")
        model = DecisionTreeRegressor(
            criterion='absolute_error', 
            splitter='best', 
            max_depth=35, 
            min_samples_split=16, 
            min_samples_leaf=15, 
            max_features='sqrt', 
            random_state=42)

    # RF retuned on RMSE
    if input_model == "_RFRetuned":
        print("Training Retuned RF model")
        model = RandomForestRegressor(
            n_estimators=600, 
            criterion='squared_error', 
            min_samples_split=18, 
            min_samples_leaf=3, 
            max_depth=35, 
            max_features=0.8, 
            bootstrap=True, 
            random_state=42)

    # fit the model on the training data
    model.fit(X_train, y_train)

    # compute the training set mean squared error
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    print(f"training set mean squared error: {train_mse}")
    
    # compute the training set root mean squared error
    train_rmse = float(np.sqrt(train_mse))
    print(f"training set rmse: {train_rmse}")
    
    # compute the test set mean squared error
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"test set mean squared error: {test_mse}")
    
    # compute the test set root mean squared error
    test_rmse = float(np.sqrt(test_mse))
    print(f"test set rmse: {test_rmse}")
    
    # save the trained model to disk
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}\n")

if __name__ == "__main__":
    
    # main(fraction=0.5) # call for hyperparameter tuning
    
    # call for model training
    models = ["_RFRetuned", "", "_GBR", "_DT"]
    for model in models:
        for fraction in [0.0, 0.25, 0.50, 0.75, 1.0]:
            main(fraction=fraction, input_model=model)

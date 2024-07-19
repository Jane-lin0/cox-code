from data_generation import generate_simulated_data

G = 5
p = 100
N_train = [200]*G
N_test = [500]*G
B_type = 1
Correlation_type = "band1"

train_data, test_data, B = generate_simulated_data(p, N_train, N_test,
                                                   B_type=B_type, Correlation_type=Correlation_type, seed=0)
X, Y, delta = train_data['X'], train_data['Y'], train_data['delta']
X_test, Y_test, delta_test = test_data['X'], test_data['Y'], test_data['delta']


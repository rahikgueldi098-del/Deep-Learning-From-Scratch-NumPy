import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(A):
    return A * (1 - A)


def forward_propagation(X, params):
    cache = {'A0' : X}
    L =len(params) // 2

    for i in range(1, L+1):
        z = params['W' + str(i)].dot(cache["A" + str(i-1)]) + params[ 'b' + str(i)]
        cache['A' + str(i)] = sigmoid(z)

    return cache

def compute_gradients(cache, y, params):

    m = y.shape[1]
    L =len(params) // 2
    dz = cache['A' + str(L)] - y
    gradients={}

    for i in reversed(range(1, L + 1)):
          gradients['dW' + str(i)]=1/m*np.dot(dz, cache['A' + str(i-1)].T)
          gradients['db' + str(i)]=1/m*np.sum(dz, axis=1, keepdims=True)
          dz=np.dot(params['W' + str(i)].T, dz)*derivative_sigmoid(cache['A' + str(i-1)])
    return gradients

def backward_propagation(X, y, params, cache, learning_rate):
    gradients= compute_gradients(cache, y, params)
    L= len(params) // 2

    for i in range(1, L + 1):
      params['W' + str(i)] = params['W' + str(i)] - learning_rate*  gradients['dW'+ str(i)]
      params['b' + str(i)] = params['b' + str(i)] - learning_rate * gradients['db' + str(i)]
    return params
if __name__ == "__main__":

    np.random.seed(42)

    X = np.random.randn(2, 3)
    y = np.random.randint(0, 2, (1, 3))


    params = {
        'W1': np.random.randn(3, 2),
        'b1': np.zeros((3, 1)),
        'W2': np.random.randn(1, 3),
        'b2': np.zeros((1, 1))
    }

    print("=== DÉBUT DU TEST ===\n")
    print("1. Poids 'W2' avant l'entraînement :")
    print(params['W2'])
    print("-" * 30)

    cache = forward_propagation(X, params)
    print("2. Sortie de la Forward Propagation (Prédictions A2) :")
    print(cache['A2'])
    print("   (Ce sont les probabilités prédites avant tout apprentissage)")
    print("-" * 30)


    learning_rate = 0.1
    params_mis_a_jour = backward_propagation(X, y, params, cache, learning_rate)

    print("3. Poids 'W2' APRÈS une itération d'entraînement :")
    print(params_mis_a_jour['W2'])
    print("   (Remarquez que les valeurs ont légèrement changé pour s'adapter aux données !)")
    print("\n=== FIN DU TEST ===")
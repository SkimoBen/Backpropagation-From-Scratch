
# Training Example 1
x = [0.1, 0.2] # Input  
target_output_value = [0.2] # Target value


# Hidden layer
input_l1_edge_weights = [0.9,0.3,0.7,0.8,0.9,0.5]
l1_biases = [0.1,0.1,0.1]

l1_l2_edge_weights = [0.1,0.5,0.78]

# Output layer
output_bias = [0.1] 

layers = 2 
# Learning rate
ß = 0.05



#SSE function.
def sum_of_squared_errors(predictions: list, targets: list) -> float: 
    se = []
    for i in range(len(predictions)): 
        x = pow(predictions[i]-targets[i], 2) 
        se.append(x)
    SSE = 0.5 * sum(se)
    return SSE

def sigmoid_activation(input, weights, biases) -> list:
    connections_per_neuron = int(len(weights) / len(biases)) # in a dense layer.
    #print(f"edges per neuron: {connections_per_neuron}")

    edge_index = 0
    neurons = len(biases)
    neuron_outputs = []
    e = 2.71828

    for neuron in range(neurons): 
        edge_output = []
        #print(f"Neuron number: {neuron}")
        b = biases[neuron]
        for edge in range(connections_per_neuron): 
            w = weights[edge_index] # Wi 
            x = input[edge] # Xi 
            z = (w*x) # Layer-wise function
            #print(f"z = {z} = weight:{w} * input:{x}")
            edge_output.append(z)
            edge_index +=1

        z = sum(edge_output) + b # Layer-wise function
        a = 1 / (1+ (pow(e,-z))) # Sigmoid activation function to make output
        neuron_outputs.append(a)

    return neuron_outputs


### Calculating the gradient
### Error term of neuron_L_j 

# can be used with all neurons
def sigmoid_derivative(sigmoid_output: float) -> float: 
    """
    Calculate the derivative of a sigmoid activation. Used on all neurons with sigmoid activations 

    Args:
        sigma (float): The sigmoid activation function output.

    Returns:
        float: The derivative of the activation
    """
    return sigmoid_output * (1-sigmoid_output)


def hadamard_sum(previos_neuron_E: list, neuron_w: list ) -> float:
    """
    Calculate the Hadamard product of a set of neuron error terms and their weights.

    This function should be used on each neuron in a hidden layer.

    Args:
        previous_neuron_e (list): The error terms of the neurons connected to the current neuron of interest.
        neuron_w (list): The edge weights connecting the current neuron to the previous layer's neurons.

    Returns:
        float: The sum of the hadamard product
    """
    hadamard = []
    for neuron in range(len(neuron_w)):
        x = previos_neuron_E[neuron] * neuron_w[neuron]
        hadamard.append(x)
    return sum(hadamard)


"""
STEPS: 
1. Run the NN feed forward. Record the activations of each neuron.
2. Get the output MSE. 
3. Calculate the output error term (the derivative of the MSE) = E 
4. 

"""

def feed_forward() -> tuple[list, list, float]:
    hidden_layer_output = sigmoid_activation(input=x, weights=input_l1_edge_weights, biases=l1_biases)
    #print(f"Hidden layer output: {hidden_layer_output}")

    final_output = sigmoid_activation(input=hidden_layer_output, weights=l1_l2_edge_weights, biases=output_bias)
    #print(f"Output Value: {final_output}")

    sse = sum_of_squared_errors(predictions=final_output, targets=target_output_value)
    return hidden_layer_output, final_output, sse


def output_neuron_derivative(output_activation_derivative: callable, output): 
    return output_activation_derivative(output)


def calc_output_weight_gradiens(last_h_layer_activations: list, output_E_term: float) -> list: 
    """
    Calculate the partial derivatives (gradients) of the output layers edge weights.

    Args:
        last_h_layer_activations (list): The outputs from the last hidden layer.
        output_E_term (float):  the error term of the output neuron.

    Returns:
        list: List of gradients
    """
    partial_derivatives = []
    for a in range(len(last_h_layer_activations)):
        partial_derivative = output_E_term * last_h_layer_activations[a]
        partial_derivatives.append(partial_derivative)
    return partial_derivatives

def update_weights(weights:list, learning_rate: float, gradients: list) -> list:
    """
    Update the weights of any given set of edges

    Args:
        weights (list): The outputs from the last hidden layer.
        learning_rate (float):  the error term of the output neuron.
        gradients (list): The gradients associated with the set of weights.

    Returns:
        list: Updated set of weights
    """
    new_weights = []
    for i in range(len(weights)):
        new_weight = weights[i] - (learning_rate * gradients[i])
        new_weights.append(new_weight)
    return new_weights

def update_biases(biases:list, learning_rate: float, gradients: list) -> list:
    """
    Update the weights of any given set of edges

    Args:
        biases (list): The outputs from the last hidden layer.
        learning_rate (float):  the error term of the output neuron.
        gradients (list): The gradients associated with the set of biases.

    Returns:
        list: Updated set of biases
    """

    new_biases = []
    for i in range(len(biases)):
        new_bias = biases[i] - (learning_rate * gradients[i])
        new_biases.append(new_bias)
    return new_biases

def backpropagate() -> tuple[list, list, float, float]:
    hidden_layer_output, final_output, sse = feed_forward()

    # g_Zk is the derivative of the output activation function
    g_Zk = output_neuron_derivative(sigmoid_derivative, final_output[0])
    #print(f"output: {final_output}, output derivative: {g_Zk}")

    # Partial derivative of the error function with respect to network output
    # Since I'm using SSE, its just the prediction - target
    network_output_error = final_output[0] - target_output_value[0]

    # This is the output neurons error term. 
    øk = network_output_error * g_Zk

    output_edge_gradients = calc_output_weight_gradiens(last_h_layer_activations=hidden_layer_output, output_E_term=øk)

    # output bias gradient is just the error signal, it's not affected by the feed forward signal 
    output_bias_gradients = øk

    new_output_weights = update_weights(weights=l1_l2_edge_weights, learning_rate=ß, gradients=output_edge_gradients)

    new_biases = update_biases(biases=output_bias, learning_rate=ß, gradients=[øk])
    return new_output_weights, new_biases, sse, final_output

    print(f"SSE: {sse}")
    # print(f"Final output: {final_output[0]} - target: {target_output_value[0]} = {final_output[0]-target_output_value[0]}")

    # print(f"øk: {øk}")
    # print(f"output bias gradients: {output_bias_gradients}")
    # print(f"Current output edge weights: {l1_l2_edge_weights}")
    # print(f"New edge weights: {new_output_weights}")
    # print(f"Current Bias: {output_bias}")
    # print(f"New Bias: {new_biases}") 

hidden_layer_output, starting_output, sse = feed_forward()
final_output=0
steps = 0
while sse > 0.01:
#for i in range(100):
    new_weights, new_biases, sse, final_output = backpropagate()
    l1_l2_edge_weights = new_weights
    output_bias = new_biases
    print(f"edges: {l1_l2_edge_weights}")
    steps +=1
    #print(f"final output: {final_output}")

print(f"Starting Output: {starting_output}")
print(f"final output: {final_output}")
print(f"SSE: {sse}")
print(f"Steps: {steps}")


def sum_of_squared_errors(predictions: list, targets: list) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) between predictions and targets.

    The SSE is calculated as the sum of the squared differences between each 
    predicted value and the corresponding target value, multiplied by 0.5.

    Args:
        predictions (list): A list of predicted values.
        targets (list): A list of target (actual) values.

    Returns:
        float: The computed Sum of Squared Errors (SSE).
        
    Example:
        predictions = [1.0, 2.0, 3.0]
        targets = [1.5, 2.0, 2.5]
        sse = sum_of_squared_errors(predictions, targets)
        print(sse)  # Output will be 0.25
    """
    se = []
    for i in range(len(predictions)):
        x = pow(predictions[i][0] - targets[i][0], 2)
        se.append(x)
    SSE = 0.5 * sum(se)
    return SSE

def layerWise_sigmoid_activation(inputs: list, weights: list, biases: list) -> list:
    """
    Perform sigmoid activation on an entire layer of neurons and add the biases

    Args:
        inputs (list): Inputs to the activation. Often called Zx
        weights (list): Edge weights of the layer 

    Returns:
    """
    connections_per_neuron = int(len(weights) / len(biases)) # in a dense layer.
    #print(f"edges per neuron: {connections_per_neuron}")
    # print(f"---Sigmoid Activation---")
    # print(f"inputs: {inputs}")
    # print(f"Edge Weights: {weights}")
    # print(f"Biases: {biases}")
    # print(f"------------------------")
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
            x = inputs[edge] # Xi 
            z = (w*x) # Layer-wise function
            #print(f"z = {z} = weight:{w} * input:{x}")
            edge_output.append(z)
            edge_index +=1

        z = sum(edge_output) + b # Layer-wise function
        a = 1 / (1+ (pow(e,-z))) # Sigmoid activation function to make output
        neuron_outputs.append(a)

    return neuron_outputs

def sigmoid_derivative(sigmoid_output: float) -> float: 

    """
    Calculate the derivative of a sigmoid activation. Used on all neurons with sigmoid activations 

    Args:
        sigmoid_output (float): The sigmoid activation function output.

    Returns:
        float: The derivative of the activation
    """
    return sigmoid_output * (1-sigmoid_output)

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

def output_neuron_derivative(output_activation_derivative: callable, output: float):
    """
    Calculate the derivative of the output neuron using the provided activation function's derivative.

    This function applies the provided activation function's derivative to the output of the neuron.

    Args:
        output_activation_derivative (callable): A function that computes the derivative of the activation function.
        output: The output value of the neuron, which will be passed to the derivative function.

    Returns:
        The result of applying the activation function's derivative to the neuron output.

    Example:
        def sigmoid_derivative(x):
            return x * (1 - x)
        
        output = 0.5
        derivative = output_neuron_derivative(sigmoid_derivative, output)
        print(derivative)  # Output will be the result of the sigmoid derivative at 0.5
    """
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


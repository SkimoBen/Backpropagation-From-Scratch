import training_data
import helpers as h

class Model: 
    def __init__(self):
        self.inputs = []
        self.EDGES_input_to_h1 = []
        self.GRADIENT_w_input_to_h1 = []
        self.GRADIENT_b_input_to_h1 = []
        self.BIASES_h1 = []
        self.GRAPH_h1 = []
        self.Ø_h1 = []
        self.FF_OUT_h1 = []
        self.EDGES_h1_to_output = []
        self.BIASES_output = []
        self.Ø_output = []
        self.hidden_layers = 0

    def feed_forward(self) -> tuple[list, list]:
        hidden_layer_output = h.layerWise_sigmoid_activation(inputs=self.inputs, weights=self.EDGES_input_to_h1, biases=self.BIASES_h1)
        model_output = h.layerWise_sigmoid_activation(inputs=hidden_layer_output, weights=self.EDGES_h1_to_output, biases=self.BIASES_output)
        #print(f"Feed Forward Model Output: {model_output}")
        return hidden_layer_output, model_output


model = Model()
model.inputs = [0.0,0.0]
model.EDGES_input_to_h1 = [0.1,0.6,0.23,0.9,0.34,0.2,0.6,0.4,0.5,0.2,0.3,0.5] # 12 input edges
model.BIASES_h1 = [0.1,0.1,0.1,0.1,0.1,0.1] # 6 hidden neurons
model.EDGES_h1_to_output = [0.1,0.3,0.4,0.2,0.7,0.4]
model.BIASES_output = [0.1]
model.hidden_layers = 1
model.Ø_output = [0.1]
model.Ø_h1 = [0,0,0,0,0,0]



data = training_data.training_examples

def process_training_examples(data: dict, model:Model) -> tuple[float, list]: 
    """
    Process the training examples using the given model and calculate the Sum of Squared Errors (SSE).

    This function iterates over the provided training data, sets the model's inputs, computes the 
    model's predictions using feedforward, and then compares the predictions to the targets. It finally 
    calculates the SSE between the predictions and the targets.

    Args:
        data (dict): A dictionary containing the training examples. Each key corresponds to a training example, 
                     and the value is a dictionary with "input" and "target" keys.
        model (Model): The model that will process the inputs to generate predictions.

    Returns:
        tuple:
            - SSE (float): The computed Sum of Squared Errors (SSE) between the model's predictions and the targets.
            - Predictions (list): The model's output predictions.
    Example:
        data = {
            1: {"input": [0.1, 0.2], "target": [0.2]},
            2: {"input": [0.3, 0.4], "target": [0.5]}
        }
        model = Model()
        sse = process_training_examples(data, model)
        print(sse)  # Output will be the calculated SSE value
    """
    predictions = []
    targets = []
    for i in range(len(data)):
        model.inputs = data[i+1]["input"]
        hidden_layer_output, model_output = model.feed_forward() 
        predictions.append(model_output)
        targets.append(data[i+1]["target"])
    #print(f"Predictions: {predictions}")
    #print(f"Targets: {targets}")
    sse = h.sum_of_squared_errors(predictions=predictions, targets=targets)
    return sse, predictions


def output_backpropagate(ß: float, data_index: int, data: dict, model: Model) -> tuple[list, list, list]:
    model.FF_OUT_h1, model_output = model.feed_forward()
    #print(f"Backprop Index: {index}")
    # g_Zk is the derivative of the output activation function
    g_Zk = h.output_neuron_derivative(h.sigmoid_derivative, model_output[0])
    #print(f"output: {final_output}, output derivative: {g_Zk}")

    # Partial derivative of the error function with respect to network output
    # Since I'm using SSE, its just the prediction - target
    #print(f"model_output[index]: {model_output[0]}")
    #print(f"data[index+1][target]: {data[index+1]['target']}")

    model_output_error = model_output[0] - data[data_index+1]["target"][0]

    # This is the output neurons error term. 
    øk = model_output_error * g_Zk
    #print(f"øk: {øk}")
    output_edge_gradients = h.calc_output_weight_gradiens(last_h_layer_activations=model.FF_OUT_h1, output_E_term=øk)

    # output bias gradient is just the error signal, it's not affected by the feed forward signal 
    model.Ø_output[0] = øk

    new_output_weights = h.update_weights(weights=model.EDGES_h1_to_output, learning_rate=ß, gradients=output_edge_gradients)

    new_biases = h.update_biases(biases=model.BIASES_output, learning_rate=ß, gradients=[øk])


    return new_output_weights, new_biases, model_output


def hidden_backpropagation(ß: float, model: Model) -> tuple[list, list]:
    #Loop through the hidden layers
    for layer in range(model.hidden_layers):
        #Loop through the hidden layer neurons
        hidden_neurons = len(model.BIASES_h1)
        for neuron in range(hidden_neurons):
            weighted_errors = [] 
            # Loop through the next layers neurons(the output layer only has 1 neuron)
            for o_neuron in range(len(model.BIASES_output)): 
                weighted_error = model.Ø_output[o_neuron] * model.EDGES_h1_to_output[neuron]
                weighted_errors.append(weighted_error)
            x = sum(weighted_errors)
            activation_derivative = h.sigmoid_derivative(sigmoid_output=model.FF_OUT_h1[neuron])
            Ø_neuron = activation_derivative * x 
            model.Ø_h1[neuron] = Ø_neuron

        edge_index = 0
        # Now deal with the weights. Loop through inputs * hiddens = edges 
        new_weights = []
        for i_neuron in range(len(model.inputs)): 
            for h_neuron in range(hidden_neurons): 
                ai = model.inputs[i_neuron]
                ø = model.Ø_h1[h_neuron] 
                w_gradient = ø*ai 
                new_weight = model.EDGES_input_to_h1[edge_index] - ß * w_gradient
                new_weights.append(new_weight)
                edge_index+=1

        new_biases = []
        for h_neuron in range(hidden_neurons): 
            new_bias = model.BIASES_h1[h_neuron] - ß * model.Ø_h1[h_neuron]
            new_biases.append(new_bias)
    return new_weights, new_biases

def stochastic_train(): 
    ß = 1.5 # Learning rate
    sse, starting_output = process_training_examples(data=data, model=model)
    starting_sse = sse
    outer_steps = 0
    inner_steps = 0
    while sse > 0.0007:
        for i in range(len(data)):
            model.inputs = data[i+1]["input"]
            new_output_weights, new_biases, model_output = output_backpropagate(ß=ß, data_index=i,data=data, model=model)
            model.EDGES_h1_to_output = new_output_weights
            model.BIASES_output = new_biases
            new_h_weights, new_h_biases = hidden_backpropagation(ß=ß, data_index=i, layer_index=1, data=data, model=model)
            model.EDGES_input_to_h1 = new_h_weights
            model.BIASES_h1 = new_h_biases

            inner_steps +=1
        sse, current_output = process_training_examples(data=data, model=model)
        print(f"SSE: {sse}")
        #print(model.BIASES_h1)
        outer_steps +=1
    print("-----")
    print(f"Starting Output; {starting_output}, Starting SSE: {starting_sse}")
    print(f"Final Output: {current_output}")
    print("-----")
    print(f"{outer_steps} Outer Steps, {inner_steps} Inner Steps")
    print("-----")
def test_model(): 
    total_difference = 0
    for i in range(len(data)):
        model.inputs = data[i+1]["input"] 
        hidden_layer_output, model_output = model.feed_forward()
        target_output = data[i+1]["target"]
        difference = round(abs((model_output[0] - target_output[0])*100),2)
        total_difference += round(difference,2)
        print(f"Input: {model.inputs}, Target Output: {target_output}, Real Output: {model_output}, Difference: {difference}%")
    print("----")
    print( f"Total Difference = {total_difference}")
    print("----")
def save_model():
    l1_edges = model.EDGES_input_to_h1
    l2_edges = model.EDGES_h1_to_output
    l1_bias = model.BIASES_h1
    l2_bias = model.BIASES_output 
    return l1_edges, l2_edges, l1_bias, l2_bias 


starting_l1_edges, starting_l2_edges, starting_l1_bias, starting_l2_bias = save_model()

stochastic_train()
test_model()
ending_l1_edges, ending_l2_edges, ending_l1_bias, ending_l2_bias = save_model()

print("Starting L1 Edges:", starting_l1_edges)
print("Ending L1 Edges:  ", ending_l1_edges)
print("Starting L2 Edges:", starting_l2_edges)
print("Ending L2 Edges:", ending_l2_edges)
print("Starting L1 Bias:", starting_l1_bias)
print("Ending L1 Bias:", ending_l1_bias)
print("Starting L2 Bias:", starting_l2_bias)
print("Ending L2 Bias:", ending_l2_bias)

#TODO 
# Status: This is a successful backpropagation algorithm for training a small model. 
# Implement back propagation for the hidden biases and input -> hidden layer weights. 
#
#













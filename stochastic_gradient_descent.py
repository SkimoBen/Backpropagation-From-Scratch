#  Stochastic gradient descent on a simple set of values 

# 1. Define the Objective Function: MSE 
# 2. Compute the Gradient (derivative) 
# 3. Set the Update Rule: r+1 = rt - h * derivative

targets = [0.9, 0.5, 0.7, 0.5,0.3,0.87,0.47]

predictions = [0,0,0,0,0,0,0]

learning_rate = 0.2

def mean_squared_error(targets, current) -> float: 
    errors = []
    L = len(current)
    for i in range(L): 
        y = targets[i] # target value
        p = current[i] # current prediction
        error = pow(y-p,2) # Squared error
        errors.append(error) # add current error to list 
    average_mse = (1/L) * sum(errors)
    return average_mse

def mse_stochastic_gradient(targets, current):
    L = len(current)
    for i in range(L): 
        y = targets[i] 
        p = current[i] 
        mse = pow(y-p,2)
        mse_gradient = -2*(y-p) # MSE Derivative
        new_p = p - learning_rate*mse_gradient # Use update rule
        predictions[i] = new_p # update the vector


starting_mse = mean_squared_error(targets=targets, current=predictions)
mse = starting_mse
i = 0
while mse > 0.00000005: 
    mse_stochastic_gradient(targets=targets, current=predictions)
    mse = mean_squared_error(targets=targets, current=predictions)
    print(f"Average MSE: {mse}")
    i+=1

mse = mean_squared_error(targets=targets, current=predictions)
print(f"Number of Iterations: {i}")
print(f"Final Predictions {predictions}")
print(f"starting MSE: {starting_mse}")
print(f"Final Avg. MSE: {mse}")
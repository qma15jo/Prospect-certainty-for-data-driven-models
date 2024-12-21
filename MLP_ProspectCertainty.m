function MLP_ProspectCertainty(hidden_size, output_masks_size, masks_ratio)

% Kindly follow the latest Release... This code was developed from scratch
% to illustrate the Prospect Certainty method for data-driven models. It
% features a simple Multi-Layer Perceptron (MLP) with a randomly generated
% dataset. The final results reflect the model's simplicity and the
% dataset's lack of coherence. However, this code is intended solely to
% facilitate the reproducibility of the method.
% 
% If you utilize this code, please cite the following paper:
% 
% Qais Yousef, Pu Li. Prospect certainty for data-driven models, 29 March
% 2024, PREPRINT (Version 1) available at Research Square
% [https://doi.org/10.21203/rs.3.rs-4114659/v1]
% 
% Additionally, please note that a comprehensive, tested package will be
% released soon.
% 
% Qais Yousef 
% 21.12.2024


% Get user input for parameters
% Example:
% MLP_ProspectCertainty(6, 3, 0.5)


% Get user input for parameters
input_size = 2; % Number of input features
% Define other constants
output_size = 1; % Number of output neurons (original)
learning_rate = 0.01; % Learning rate
num_epochs = 2000; % Number of training epochs


%% Initialize weights and biases
rng('default'); % For reproducibility
W1 = randn(hidden_size, input_size) * sqrt(2 / input_size); % Initialization
b1 = zeros(hidden_size, 1);
W2 = randn(output_size, hidden_size) * sqrt(2 / hidden_size);
b2 = zeros(output_size, 1);

% New Mask neurons partially connected weights and biases
W2_masks = randn(output_masks_size, hidden_size) * sqrt(2 / hidden_size);
b2_masks = zeros(output_masks_size, 1);

% Create a mask for partial connections
connection_mask = rand(output_masks_size, hidden_size) < masks_ratio;

% Define activation functions
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivative = @(x) x .* (1 - x);



%% Generate synthetic data
% A random generation for two inputs and one output. This can be replace
% with the user's synthetic dataset.
num_samples = 100;
class1 = [randn(num_samples, 2) + 1; randn(num_samples, 2) - 1]; % Class 1
class2 = [randn(num_samples, 2) - 1; randn(num_samples, 2) + 1]; % Class 2

% Combine the data
X = [class1; class2];
y = [ones(2 * num_samples, 1); zeros(2 * num_samples, 1)]; % Labels: 1 for class 1, 0 for class 2
y = y'; % Transpose for easier matrix operations



%% Training loop
for epoch = 1:num_epochs
    % Forward propagation
    Z1 = W1 * X' + b1;           % Hidden layer linear combination
    A1 = sigmoid(Z1);            % Hidden layer activation
    Z2 = W2 * A1 + b2;           % Original output neuron linear combination
    A2 = sigmoid(Z2);            % Original output neuron activation (prediction)
    
    % Mask neurons forward propagation with partial connections
    Z2_masks = (W2_masks .* connection_mask) * A1 + b2_masks; % Mask neurons linear combination
    A2_masks = sigmoid(Z2_masks);        % Mask neurons activation
    
    % Final output is the average of the original and Mask neurons
    A2_final = (A2 + sum(A2_masks, 1)) / (1 + output_masks_size);
    
    % Compute loss (binary cross-entropy)
    m = size(X, 1); % Number of samples
    loss = -(1 / m) * sum(y .* log(A2_final) + (1 - y) .* log(1 - A2_final));
    
    % Backward propagation for original output neuron
    dZ2 = A2_final - y;               % Output layer error
    dW2 = (1 / m) * dZ2 * A1';        % Gradient for W2
    db2 = (1 / m) * sum(dZ2, 2);      % Gradient for b2
    
    % Backward propagation for Mask neurons with partial connections
    dZ2_masks = repmat(dZ2 / (1 + output_masks_size), output_masks_size, 1); % Mask neurons error
    dW2_masks = (1 / m) * (dZ2_masks * A1'); % Gradient for W2_Mask with mask applied
    dW2_masks = dW2_masks .* connection_mask; % Apply mask to gradients
    db2_masks = (1 / m) * sum(dZ2_masks, 2); % Gradient for b2_Mask
    
    % Backward propagation for hidden layer
    dA1 = W2' * dZ2 + (W2_masks' .* connection_mask') * dZ2_masks; % Combine gradients from original and Mask neurons
    dZ1 = dA1 .* sigmoid_derivative(A1);     % Apply derivative of activation function
    dW1 = (1 / m) * dZ1 * X;                 % Gradient for W1
    db1 = (1 / m) * sum(dZ1, 2);             % Gradient for b1
    
    % Update weights and biases
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W2_masks = W2_masks - learning_rate * dW2_masks;
    b2_masks = b2_masks - learning_rate * db2_masks;
    
    % Print loss every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
    end
end



%% Make predictions
Z1 = W1 * X' + b1;
A1 = sigmoid(Z1);
Z2 = W2 * A1 + b2;
Z2_masks = (W2_masks .* connection_mask) * A1 + b2_masks;

[best_alternative_idx_vector] = MainProspectCertainty(Z2, Z2_masks);

% Combine the logit and its masks to select the refined value. (A) for the logit.
Z2_Z2_masks = [Z2;Z2_masks];
% Generate linear indices 
lin_indices = sub2ind(size(Z2_Z2_masks), best_alternative_idx_vector, 1:size(Z2_Z2_masks,2)); 
% Select the values using linear indices 
refined_values = Z2_Z2_masks(lin_indices);

% A2 = sigmoid(Z2);
% A2_masks = sigmoid(Z2_masks);
% A2_final = (A2 + sum(A2_masks, 1)) / (1 + output_masks_size);
A2_final = sigmoid(refined_values);
predictions = A2_final > 0.5; % Convert probabilities to class labels

% Calculate accuracy
accuracy = sum(predictions == y) / length(y) * 100;
fprintf('Classification accuracy: %.2f%%\n', accuracy);




%% Visualize the neurons and connections
figure;
hold on;

% Plot the input layer neurons
input_layer_neurons = input_size;
for i = 1:input_layer_neurons
    plot([0, 1], [i, input_layer_neurons/2 + 0.5], 'ko-');
end

% Plot the hidden layer neurons and connections
hidden_layer_neurons = hidden_size;
for i = 1:hidden_layer_neurons
    plot([1, 2], [input_layer_neurons/2 + 0.5, i], 'ko-');
end

% Plot the original output neuron and connections
output_layer_neurons = 1;
y_position_original_output = hidden_layer_neurons / 2 + 0.5; % Original output neuron y-position
for i = 1:hidden_layer_neurons
    plot([2, 3], [i, y_position_original_output], 'ko-');
end
plot([3, 3], [y_position_original_output, y_position_original_output], 'ko-');
text(3.5, y_position_original_output, 'original neuron (A)', 'HorizontalAlignment', 'center');

% Plot the Mask neurons and partial connections
for j = 1:output_masks_size
    y_position_masks = y_position_original_output - (j * 0.5); % Position Mask neurons under the original neuron
    for i = 1:hidden_layer_neurons
        if connection_mask(j, i)
            plot([2, 3], [i, y_position_masks], 'ko-');
        end
    end
    plot([3, 3], [y_position_masks, y_position_masks], 'ko-');
    text(3.5, y_position_masks, 'mask neuron', 'HorizontalAlignment', 'center');
end

% Labels and title
text(0, input_layer_neurons + 1, 'Input Layer', 'HorizontalAlignment', 'center');
text(1, input_layer_neurons + 1, 'Hidden Layer', 'HorizontalAlignment', 'center');
text(3, hidden_layer_neurons, 'Output Layer', 'HorizontalAlignment', 'center');
title('MLP Neurons and Connections with the Generated Masks');
axis([0 3.5 0 hidden_layer_neurons + 2]);
axis off;
hold off;



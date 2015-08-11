function [J, grad] = nnCostFunction(nn_params, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, ...
X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Feedforward Propogation
a1 = [ones(m, 1), X]; % results in [5000, 401]
z2 = Theta1 * a1'; % results in [25, 5000]% Hidden Layer
a2 = sigmoid(z2);  % results in [25, 5000]
a2 = [ones(1, size(a2, 2)); a2]; % results in [26, 5000]% Output layer
z3 = Theta2 * a2; % results in [10, 5000]
a3 = sigmoid(z3); % results in [10, 5000]


% Get cost function
% Map the y vector to a matrix
y_matrix = zeros(num_labels, m); % y_matrix [10, 5000]
for i = 1:m
    y_matrix(y(i), i) = 1;
end
% y is [5000, 1]
for i = 1 : m
    J += sum(-y_matrix'(i, :) .* log(a3'(i,:)) - (1-y_matrix')(i,:) .* log(1-a3'(i,:))); 
end
J = J / m;
regular_1 = 0;
regular_2 = 0;
regular_1 = Theta1(:, 2:end) .^ 2;
regular_2 = Theta2(:, 2:end) .^ 2;
J = J + lambda*(sum(sum(regular_1)) + sum(sum(regular_2))) / (2*m);

% the following are the code of gradients
% since y is [5000, 1] vector, it must converted
% a3 is [10, 5000]
% y_matrix is [10, 5000]

Delta1 = zeros(size(Theta1)); 
Delta2 = zeros(size(Theta2));

for i = 1:m
    delta3 = a3(:,i) - y_matrix( :,i);
    mid_1 = (Theta2)' * delta3;
    mid_2 = sigmoidGradient(z2( :,i));
    % size(mid_1);
    % size(mid_2);
    delta2 = ((Theta2)' * delta3)(2:end,: ) .* sigmoidGradient(z2( :,i));
    % size(delta2);

    Delta2 = Delta2 + delta3*(a2( :,i))';
    Delta1 = Delta1 + delta2*a1(i,: );
    % size(Delta2);
end
    % size(Delta2);
Theta2_grad = Delta2 / m;
Theta1_grad = Delta1 / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

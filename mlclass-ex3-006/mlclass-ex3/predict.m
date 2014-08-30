function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);   % 5000 in this case
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];
z2 = zeros(m, size(Theta1, 1)); 

for i = 1:m
	z2(i, :) = Theta1 * X(i, :)';  % m * 25
end
	
z2 = [ones(m, 1) sigmoid(z2)]; % m * (25 + 1)

for j = 1:m
	z3(j, :) = Theta2 * z2(j, :)'; % m * num_labels
end

z3 = sigmoid(z3);

for k = 1:m
	p(k) = find(z3(k, :) == max(z3(k,:)));
end

% =========================================================================


end

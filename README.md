#MLAndrewNg
==========
##Programming Lesson 2

==========
###CostFunction

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples (100)

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i = 1 : m
	J = J + ( - y(i) * log(sigmoid(X(i, :) * theta)) - (1 - y(i)) * log(1 - sigmoid(X(i, :) * theta)));
endfor

J = J ./ m;

for j = 1 : size(theta)
	part = 0;

	for i = 1 : m
		part = part + (sigmoid(X(i, :) * theta) - y(i)) * X(i, j);
	endfor
	

	grad(j) = part ./ m;
endfor

% =============================================================

end

===========
###sigmoid 

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1 ./ (1 + exp(-z));

% =============================================================

end

===============
###predict
function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
for i = 1: m
	if  X(i, :) * theta >= 0
		p(i) = 1;
	else 
		p(i) = 0;
	end
end 
% =========================================================================
end

===================
### costfnReg

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1 : m 
	J = J + (- y(i) * log (sigmoid(X(i,:)* theta)) - (1 - y(i)) * log(1 - sigmoid(X(i,:)*theta))) / m ;
end

J = lambda * theta'(1, 2:size(theta)) * theta(2:size(theta),1) / (2 * m) + J ; 

for i = 1 : m
	grad(1) = grad(1) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,1);
end

grad(1) = grad(1) / m;

for j = 2 : size(theta)

	for i = 1 : m
	grad(j) = grad(j) + (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
	end
	
	grad(j) = (grad(j) / m) + lambda * theta(j) / m;
end

% =============================================================

end




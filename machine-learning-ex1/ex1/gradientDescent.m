function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	H = theta'.*X;
	H = H(:,1)+H(:,2)-y;
	H1 = H.*X(:,1);
	H2 = H.*X(:,2);
	
	theta0 = theta(1) - (alpha/m)*sum(H1);
	theta1 = theta(2) - (alpha/m)*sum(H2);
	
	theta = [theta0;theta1];


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	fprintf('\n%d - With theta0 = %f and theta1 = %f Cost computed = %f\n',iter,theta0,theta1,J_history(iter))

end

end

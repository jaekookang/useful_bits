function [J, grad] = cost_function(theta, X, y)
% This function computes cost and gradients w.r.t. each theta
% for logistic regression
% 2017-04-08 jkang
% Matlab R2016b
% ref: Machine Learning by Andrew Ng (coursera.org)
%
% **input**
%    theta: m x 1 vector
% intput X: n x m matrix (n: number of examples/observations, m: number of features)
% output y: n x 1 vector (n: number of examples/observations)
%
% **output**
%    J: cost (scalar)
% grad: m x 1 vector (gradient w.r.t. each theta)

cost_raw = -y'*log(sigmoid(X*theta)) - (1 - y)'*log(1 - sigmoid(X*theta));
J = 1/m*sum(cost_raw);

grad = 1/m*((sigmoid(X*theta)-y)'*X)';
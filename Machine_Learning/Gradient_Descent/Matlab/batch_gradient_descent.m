% batch gradient descent example 
% in univariate linear regression
% 2017-03-24 jkang
% Matlab R2016b
%
% y = th0 + th1*x

%% Training
% data
mu = [2,3];
sigma = [1,1.5;1.5,3];
n_input = 100;
r = mvnrnd(mu,sigma,n_input);

xdata = [ones(1,100);r(:,1)']; % (feature) x (example), padded with ones
ydata = r(:,2)';

% parameters
theta = [0, 0];
max_iter = 30;
alpha = 0.01;

% update
MSE = @(x,y,theta) 1/(2*length(x))*sum((theta*xdata - ydata).^2);
for i = 1:max_iter
    theta = theta - alpha*(1/n_input)*(theta*xdata - ydata)*xdata';
    fprintf('MSE=%.4f\n',MSE(xdata,ydata,theta))
end

%% Plot
plot(xdata(2,:),ydata,'o')
hold on
plot(xdata(2,:),theta*xdata,'r-')
hold off



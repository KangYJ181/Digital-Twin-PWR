% % fuction: realize GPR
clear, clc

load("A_T_FC.mat")  % POD coef. data

% mode number
num = 5 ;

% training data
X = (15:33)';
Y = A_T_FC(num,:)';

% normalization
X_norm = (X-min(X)) / (max(X)-min(X));
Y_norm = (Y-min(Y)) / (max(Y)-min(Y));
X_test_norm = (0:0.01:1)';

% optimization ranges
s_l = 0.01;  s_u = 1;
l_l = 0.01;  l_u = 0.5;
sigma_l = 0.01;  sigma_u = 0.1;

% optimization steps
ds = 0.01;
dl = 0.01;
dsigma = 0.01;

% initialization
TARGET = -1e6;
s_opt = 0;
l_opt = 0;
sigma_opt = 0;

s_process = [];
l_process = [];
sigma_process = [];
TARGET_process = [];

% grid search optimization
for s = s_l:ds:s_u
    for l = l_l:dl:l_u
        for sigma = sigma_l:dsigma:sigma_u
            target = L_THETA(X_norm,Y_norm,s,l,sigma);
            if target > TARGET
                TARGET = target;
                s_opt = s;
                l_opt = l;
                sigma_opt = sigma;
                s_process = [s_process;s_opt];
                l_process = [l_process;l_opt];
                sigma_process = [sigma_process;sigma_opt];
                TARGET_process = [TARGET_process;TARGET];
            end
        end
    end
end


function target = L_THETA(X,Y,s,l,sigma)
% Target function
% X:training input vector
% Y:training output vector
% s,l,sigma:hyperparameters
[N,~] = size(X);
target = -1/2 * Y' * inv(K_XX(X,X,s,l)+sigma^2*eye(N)) * Y...
         -1/2 * log(det(K_XX(X,X,s,l)+sigma^2*eye(N)));
end

function K = K_XX(X1,X2,s,l)
% Covariance matrix
% X1:vector 1 
% X2:vector 2
% s,l:hyperparameters
N1 = length(X1);
N2 = length(X2);
K = zeros(N1,N2);
for k1 = 1:N1
    for k2 = 1:N2
        K(k1,k2) = kernel(X1(k1),X2(k2),s,l);
    end
end
end

function k = kernel(x1,x2,s,l)
% Kernel function
% x1:variable 1
% x2:variable 2
% s,l:hyperparameters
k = s^2 * exp(-(x1-x2)^2/2/l^2);
end
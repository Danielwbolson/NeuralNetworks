function [L, dLdy] = Loss_cross_entropy_softmax(x, y)

% L = sum [ y(i) * log(y_tilde(i)) ]
% dLdy is in regards to y_tilde
% dLdy = y)i) / y_tilde(i)

L = [];
dLdy = [];
y_tilde = [];

% Calculating denominator of y_tilde func
denominator = 0;
for i = 1 : size(y)
    value = exp(x(i));
    denominator = denominator + value;
end

% Setting our y_tilde
for i = 1 : size(y)
    y_tilde = [y_tilde; exp(x(i)) / denominator];
end

% Now calculate L and dLdy
L = sum(y.*log(y_tilde));
dLdy = y./y_tilde;

end


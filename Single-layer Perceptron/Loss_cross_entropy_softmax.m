function [L, dLdy] = Loss_cross_entropy_softmax(x, y)

% L = sum [ y(i) * log(y_tilde(i)) ]
% dLdy is in regards to y_tilde
% dLdy = y(i) / y_tilde(i)

% SoftMax
y_tilde = SoftMax(x);

% Now calculate L and dLdy
L = -sum(y.*log(y_tilde));

% Derivative of y * log(y_tilde) is:
%     y * (1 / y_tilde)
dLdy = transpose(y_tilde - y);

end


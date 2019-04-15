function [dLdx, dLdw, dLdb] = FC_backward(dLdy, x, w, b, y)

% n = 196, m = 10


% L = (y - y_guess)^2
% y_guess = wx + b
%
% L = (y - wx - b)^2
% dLdy = 2(y - wx - b)
% dLdx = -2(y - wx-b)w
% dLdw = -2(y - wx - b)x
% dLdb = -2(y - wx - b)

z = -2*(y - (w*x + b));

% dLdx, 1xm
dLdx = w.* z;

% dLdb, 1xn
dLdb = z;

% dLdw, 1x(mXn)
dLdw = z * x';

end


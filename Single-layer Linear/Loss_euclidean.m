function [L, dLdy] = Loss_euclidean(y_tilde, y)

L = sum(y - y_tilde)^2;
dLdy = -2*(y - y_tilde);

end


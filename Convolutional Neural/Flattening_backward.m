function [dLdx] = Flattening_backward(dLdy, x, y)

dLdx = reshape(dLdy .* y, size(x));

end


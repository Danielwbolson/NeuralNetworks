function [dLdx, dLdw, dLdb] = FC_backward(dLdy, x, w, b, y)

% n = 196, m = 10

% dLdy = 1xm

% dLdx, 1xn
dLdx = transpose(dLdy * w);

% dLdb, 1xm
dLdb = transpose(dLdy * 1);

% dLdw, 1x(mXn
dYdw = zeros(length(dLdy), size(w,1)*size(w,2));
for i = 1:size(dYdw, 1)
    s_index = (i-1)*size(x, 1) + 1;
    end_index = s_index + size(x, 1) - 1;
    dYdw(i, (s_index:end_index)) = x(:);
end

dLdw = reshape(dLdy * dYdw, size(w));

end


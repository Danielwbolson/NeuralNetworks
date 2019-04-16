function [dLdx, dLdw, dLdb] = FC_backward(dLdy, x, w, b, y)

% n = 196, m = 10

% dLdx, 1xn
dLdx = transpose(dLdy * w);

% dLdb, 1xm
dLdb = transpose(dLdy * 1);

% dLdw, 1x(mXn)
% Create our 10 x 1960 matrix
dYdw = zeros(length(dLdy), size(w,1)*size(w,2));

a = x(:);
% Put our x (img) at intervals inside our matrix
% for each row, we move our img 196 columns to the right
% we end up with a huge "diagonal" matrix
for i = 1:size(dYdw, 1)
    s_index = (i-1)*size(x, 1) + 1;
    end_index = s_index + size(x, 1) - 1;
    dYdw(i, (s_index:end_index)) = transpose(x(:));
end

% Multiply the matrices and turn our long vector into a 10x196 matrix
dLdw = [];
wVec = dLdy * dYdw;
for i = 1:size(w, 1)
    s_index = (i-1)*size(w, 2)+1;
    end_index = i*size(w, 2);
    dLdw = [dLdw; wVec(s_index:end_index)];
end

end

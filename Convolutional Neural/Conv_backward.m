function [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y)

x_padded = conv2(x,[0,0,0;0,1,0;0,0,0]);
imgCol = im2col(x_padded, [size(w_conv, 1), size(w_conv, 2)], 'sliding');

% dLdw = 3x3x1x3
dLdw = reshape(imgCol * reshape(dLdy(:), [196, 3]), [3, 3, 1, 3]);

% dLdb = 3x1
for i = 1:size(b_conv)
    mat = dLdy(:,:,i);
    dLdb(i) = sum(mat(:));
end

end


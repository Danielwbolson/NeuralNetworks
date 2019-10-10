function [y] = Conv(x, w_conv, b_conv)

% Where to include b_conv?

x_padded = conv2(x,[0,0,0;0,1,0;0,0,0]);

imgCol = im2col(x_padded, [size(w_conv, 1), size(w_conv, 2)], 'sliding');

for i = 1:size(w_conv, 4)
    convo = w_conv(:, :, :, i);
    convo_vec = convo(:);
    y(:, :, i) = reshape(transpose(imgCol)*convo_vec, [14, 14]) + b_conv(i);
end

end


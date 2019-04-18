function [dLdx] = ReLu_backward(dLdy, x, y)

y = [];
for i = 1 : size(x, 1)
    for j = 1 : size(x, 2)
        for k = 1 : size(x, 3)
            if x(i, j, k) < 0
                y(i, j, k) = 0;
            else
                y(i, j, k) = 1;
            end
        end
    end
end

for i = 1:size(y, 3)
    dLdx(:, :, i) = transpose(dLdy(:, :, i) .* y(:, :, i));    
end

end


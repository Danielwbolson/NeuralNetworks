function [dLdx] = ReLu_backward(dLdy, x, y)

y = [];
for i = 1 : size(x, 1)
    for k = 1 : size(x, 2)
        if x(i, k) < 0
            y(i, k) = 0;
        else
            y(i, k) = 1;
        end
    end
end

dLdx = transpose(dLdy .* y);        

end


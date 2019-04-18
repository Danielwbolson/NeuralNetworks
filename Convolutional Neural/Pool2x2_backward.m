function [dLdx] = Pool12x2_backward(dLdy, x, y)

val = dLdy .* y;

dLdx = zeros(size(x, 1), size(x, 2), size(x, 3));
for i = 1:size(y, 1)
    for j = 1:size(y, 2)
        for k = 1:size(y, 3)
            if x(i*2-1, j*2-1, k) == y(i, j, k)
                dLdx(i*2-1, j*2-1, k) = val(i, j, k);
            elseif x(i*2-1, j*2, k) == y(i, j, k)
                dLdx(i*2-1, j*2, k) = val(i, j, k);
            elseif x(i*2, j*2-1, k) == y(i, j, k)
                dLdx(i*2, j*2-1, k) = val(i, j, k);
            elseif x(i*2, j*2, k) == y(i, j, k)
                dLdx(i*2, j*2, k) = val(i, j, k);
            end
        end
    end
end

end


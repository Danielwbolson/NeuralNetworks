function [y] = ReLu(x)

for i = 1:size(x, 1)
    for k = 1:size(x, 2)
        y(i, k) = max(0, x(i, k));
    end
end

end


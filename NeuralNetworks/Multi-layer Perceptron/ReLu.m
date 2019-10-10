function [y] = ReLu(x)

for i = 1:size(x, 1)
    for j = 1:size(x, 2)
        for k = 1:size(x, 3)
            y(i, j, k) = max(0, x(i, j, k));
        end
    end
end

end


function [y] = Pool2x2(x)

y = zeros(size(x, 1)/2, size(x, 2)/2, size(x, 3));

for i = 1:2:size(x, 1)
    for j = 1:2:size(x, 2)
        for k = 1:size(x, 3)
            y(floor(i/2 + 1), floor(j/2 + 1), floor(k)) = ...
                max([x(i, j, k), x(i+1, j, k), x(i, j+1, k), x(i+1, j+1, k)]);            
        end
    end
end

end


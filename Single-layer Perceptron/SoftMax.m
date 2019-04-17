function [y] = SoftMax(x)

y = [];

% Calculating denominator of y_tilde func
denominator = 0;
for i = 1 : size(x)
    value = exp(x(i));
    denominator = denominator + value;
end

% Setting our y_tilde
for i = 1 : size(x)
    y = [y; exp(x(i)) / denominator];
end

end


function [w, b] = TrainSLP(mini_batch_x, mini_batch_y)

learningRate = 2;
decayRate = 0.8;
nIters = 10000;

% Random initial weights and bias
w = rand(10, 196);
b = rand(10, 1);
k = 1;

for i = 1:nIters
    % Every 1000 iterations, multiply by decay rate
    if mod(i, 1000) == 0
        learningRate = decayRate * learningRate;
    end
    dLdw_grad = 0; dLdb_grad = 0;
    
    % For each image inside our mini_batch
    for j = 1:size(mini_batch_x{k},2)
        % Get our image and actual answer
        img = mini_batch_x{k}(:, j);
        y_actual = mini_batch_y{k}(:, j);
        
        % Calculate guessed answer
        y_guess = FC(img, w, b);
        
        % Find error
        [L, dLdy] = Loss_cross_entropy_softmax(y_guess, y_actual);
        
        % Back propogate
        [dLdx dLdw dLdb] = FC_backward(dLdy, img, w, b, y_actual);
        
        dLdw_grad = dLdw_grad + dLdw;
        dLdb_grad = dLdb_grad + dLdb;
    end
    
    % Loop over mini batches
    k = k + 1;
    if k > size(mini_batch_x, 2)
        k = 1;
    end
    
    % Update weights
    w = w - learningRate*dLdw_grad/nIters;
    b = b - learningRate*dLdb_grad/nIters;       
    
end

end


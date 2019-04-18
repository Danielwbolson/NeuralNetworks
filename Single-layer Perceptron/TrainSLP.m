
function [w, b] = TrainSLP(mini_batch_x, mini_batch_y)

learningRate = 0.05;
decayRate = 0.9;
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
    dLdw = 0; dLdb = 0;
    
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
        [dldx, dldw, dldb] = FC_backward(dLdy, img, w, b, y_actual);
        
        dLdw = dLdw + dldw;
        dLdb = dLdb + dldb;
    end
    
    % Loop over mini batches
    k = k + 1;
    if k > size(mini_batch_x, 2)
        k = 1;
    end
    
    % Update weights
    w = w - (learningRate/size(mini_batch_x{k}, 2))*dLdw;
    b = b - (learningRate/size(mini_batch_x{k}, 2))*dLdb;       
    
end

end


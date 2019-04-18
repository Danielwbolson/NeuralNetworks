function [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y)
learningRate = 0.05;
decayRate = 0.9;
nIters = 10000;

% Random initial weights and bias
w1 = rand(30, 196);
b1 = rand(30, 1);
w2 = rand(10, 30);
b2 = rand(10, 1);
k = 1;

for i = 1:nIters
    % Every 1000 iterations, multiply by decay rate
    if mod(i, 1000) == 0
        learningRate = decayRate * learningRate;
    end
    dLdw1 = 0; dLdw2 = 0; dLdb1 = 0; dLdb2 = 0;
    
    % For each image inside our mini_batch
    for j = 1:size(mini_batch_x{k},2)
        % Get our image and actual answer
        img = mini_batch_x{k}(:, j);
        y_actual = mini_batch_y{k}(:, j);
        
        % Calculate guessed answer
        a1 = FC(img, w1, b1);
        % Relu Activiation function
        f1 = ReLu(a1);
        % Second layer calculation of guess
        a2 = FC(f1, w2, b2);
        % Find error
        [L, dLdy] = Loss_cross_entropy_softmax(a2, y_actual);
        
        % Back propogate for w2 and b2
        [da2df1, da2dw2, da2db2] = FC_backward(dLdy, f1, w2, b2, a2);
        
        % Back propogate for w1 and b1
        [df1da1] = ReLu_backward(da2df1, f1, a2);
        [da1dx, da1dw1, da1db1] = FC_backward(df1da1, img, w1, w2, a1);
        
        
        dLdw1 = dLdw1 + da1dw1;
        dLdb1 = dLdb1 + da1db1;
        
        dLdw2 = dLdw2 + da2dw2;
        dLdb2 = dLdb2 + da2db2;
    end
    
    % Loop over mini batches
    k = k + 1;
    if k > size(mini_batch_x, 2)
        k = 1;
    end
    
    % Update weights
    w1 = w1 - (learningRate/size(mini_batch_x{k}, 2))*dLdw1;
    b1 = b1 - (learningRate/size(mini_batch_x{k}, 2))*dLdb1;  
    
    w2 = w2 - (learningRate/size(mini_batch_x{k}, 2))*dLdw2;
    b2 = b2 - (learningRate/size(mini_batch_x{k}, 2))*dLdb2; 
    
end

end
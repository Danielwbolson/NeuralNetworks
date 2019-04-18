
function [w_conv, b_conv, w_fc, b_fc] = TrainCNN(mini_batch_x, mini_batch_y)

learningRate = 0.05;
decayRate = 0.9;
nIters = 10000;

% Random initial weights and bias
w_conv = rand(3, 3, 1, 3);
b_conv = rand(3, 1);
w_fc = rand(10, 147);
b_fc = rand(10, 1);
k = 1;

for i = 1:nIters
    % Every 1000 iterations, multiply by decay rate
    if mod(i, 1000) == 0
        learningRate = decayRate * learningRate;
    end
    d_w_conv = 0; d_b_conv = 0; d_w_fc = 0; d_b_fc = 0;
    
    % For each image inside our mini_batch
    for j = 1:size(mini_batch_x{k},2)
        % Get our image and actual answer
        img = reshape(mini_batch_x{k}(:, j), [14, 14, 1]);
        y_actual = mini_batch_y{k}(:, j);
        
        % Convolution layer
        c = Conv(img, w_conv, b_conv);
        % Relu activiation function
        r = ReLu(c);
        % Max pool2x2 step
        p = Pool2x2(r);
        % Flatten
        f = Flattening(p);
        % Forward propogation
        fc = FC(f, w_fc, b_fc);
        % Calculate softmax, loss, and derivative
        [L, dLdy] = Loss_cross_entropy_softmax(fc, y_actual);
        
        % Back propogate for w_fc and b_fc
        [dfc_f, dfc_wfc, dfc_bfc] = FC_backward(dLdy, f, w_fc, b_fc, fc);
        
        % Back propogate for w_conv and b_conv
        [dfdp] = Flattening_backward(dfc_f, p, f);
        [dpdr] = Pool2x2_backward(dfdp, r, p);
        [drdc] = ReLu_backward(dpdr, c, r);
        [dcdw, dcdb] = Conv_backward(drdc, img, w_conv, b_conv, c);
        
        % Add up weight of all batch
        d_w_conv = d_w_conv + dcdw;
        d_b_conv = d_b_conv + dcdb;
        
        d_w_fc = d_w_fc + dfc_wfc;
        d_b_fc = d_b_fc + dfc_bfc;
    end
    
    % Loop over mini batches
    k = k + 1;
    if k > size(mini_batch_x, 2)
        k = 1;
    end
    
    % Update weights
    w_conv = w_conv - (learningRate/size(mini_batch_x{k}, 2))*d_w_conv;
    b_conv = b_conv - (learningRate/size(mini_batch_x{k}, 2))*d_b_conv;  
    
    w_fc = w_fc - (learningRate/size(mini_batch_x{k}, 2))*d_w_fc;
    b_fc = b_fc - (learningRate/size(mini_batch_x{k}, 2))*d_b_fc; 
    
end

end
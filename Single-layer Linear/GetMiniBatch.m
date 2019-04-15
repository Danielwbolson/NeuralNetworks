function [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train, label_train, batch_size)

num_batch = ceil(size(im_train, 2) / batch_size);

% Holding batches of images
% image length x batch_size
mini_batch_x = [];
% Holding batches of labels
% One-hot encoding of 10 x batch_size
mini_batch_y = [];

indices = randperm(size(im_train, 2));

for i = 1:num_batch
    images = [];
    labels = zeros(10, batch_size);
    for k = 1:batch_size
        % Get our random index to grab from
        train_index = indices((i-1)*batch_size + k);
        
        % Cache our image and label
        images = [images im_train(:, train_index)];
        labels(label_train(:, train_index)+1, k) = 1;
    end
    % Store our mini batches
    mini_batch_x{i} = images;
    mini_batch_y{i} = labels;
end

end


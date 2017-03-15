function [] = NN(iters, alpha)
    W = [ 0.6 0.7 0.0;
         0.01 0.43 0.88];
    w = [0.02;0.03; 0.09];
    X = [0.75 0.8; 0.2 0.05; -0.75 0.8; 0.2 -0.05];
    y = [1;1;-1;-1];
    
    for i = 1:iters
        disp('-----------------------------')
        disp('-----------------------------')
        disp('-----------------------------')
        disp(sprintf ('Iteration # %d', i))
        disp('Forward pass')
        %%  Forward Pass
        % hidden layer
        s = X*W % [4 x 3]
        z = tanh(s) % [ 4 x 3]
        
        % output layer
        s_out = z * w % [4 x 1]
        z_out = relu(s_out)

        loss = 0.5 * sum((z_out - y).^2);
        disp(sprintf('loss:  %f ', loss))

        %% Backpropogation
        disp('Backward pass')
        % compute deltas 
        delta_out = (z_out - y) .* der_tanh(s_out) % [4 x 1]
        delta_1 = delta_out*w' .* der_relu(s) % [4 x 3]
        
        delta_w = z' * delta_out
        delta_W = X' * delta_1

        %% Update
        disp('Updated parameters')
        w = w - alpha * z' * delta_out
        W = W - alpha * X' * delta_1
    end

end

function y = der_tanh(x)
    y = 1 - tanh(tanh(x));
end

function y = der_relu(x)
    y = x > 0;
end

function y = relu(x)
    y = max(0,x);
end
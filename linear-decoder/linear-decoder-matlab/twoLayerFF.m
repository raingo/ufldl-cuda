%% forward pass in one hidden layer NN
% input: dInput * nSamples
% W1: dHidden * dInput
% W2: dOutput * dHidden
% b1: dHidden * 1
% b2: dOutput * 1
% a2: dHidden * nSamples
% a3: dOutput * nSamples
function [a2, a3] = twoLayerFF(input, W1, W2, b1, b2)

a2 = W1 * input; % dHidden * nSamples

a2 = bsxfun(@plus, a2, b1);
a2 = sigmoid(a2);

a3 = W2 * a2;
a3 = bsxfun(@plus, a3, b2); % dOutput * nSamples

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));

end

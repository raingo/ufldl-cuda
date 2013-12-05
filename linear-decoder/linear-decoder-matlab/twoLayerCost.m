function [cost, delta, rho] = twoLayerCost(data, a2, a3, lambda, W1, W2, sparsityParam, beta)

nSamples = size(data, 2);

rho = sum(a2, 2) / nSamples;

delta = a3 - data;

cost = norm(delta(:), 2) ^ 2;
cost = cost / nSamples;

cost = cost + lambda * (norm(W1(:), 2) ^ 2 + norm(W2(:), 2) ^ 2);
cost = cost / 2;
% 
cost = cost + beta * sum(KL(rho, sparsityParam));

end

function div = KL(rho, sparsty)

div = sparsty .* log(sparsty ./ rho) + (1 - sparsty) .* log((1 - sparsty) ./ (1 - rho));

end

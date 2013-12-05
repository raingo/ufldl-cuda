%% Error backpropagation
% input: dInput * nSamples
% output: dOutput * nSamples
% W1: dHidden * dInput
% W2: dOutput * dHidden
% a2: dHidden * nSamples
% a3: dOutput * nSamples
%
% pGradW1: dHidden * dInput
% pGradW2: dOutput * dHidden
% pGradb1: dHidden * 1
% pGradb2: dOutput * 1
function [pGradW1, pGradW2, pGradb1, pGradb2] = twoLayerBP(input, delta3, W1, W2, a2, ...
        rho, beta, sparsity, lambda)

    nSamples = size(input, 2);

    sparsity_der = beta * (- sparsity ./ rho + (1 - sparsity) ./ (1 - rho));
    sparsity_der = repmat(sparsity_der, 1, nSamples);
    
    delta2 = W2' * delta3;
    delta2 = (delta2 + sparsity_der) .* a2 .* (1 - a2); % dHidden * nSamples

    pGradW2 = delta3 * a2'; % dHidden * dInput
    pGradW2 = pGradW2 / nSamples + lambda * W2;

    pGradb2 = sum(delta3, 2); % dOutput * 1
    pGradb2 = pGradb2/ nSamples;

    pGradW1 = delta2 * input'; % dOutput * dHidden
    pGradW1 = pGradW1 / nSamples + lambda * W1;

    pGradb1 = sum(delta2, 2); % dHidden * nSamples --> dHidden * 1
    pGradb1= pGradb1 / nSamples;


end

function div = KL(rhoHat, rho)

    div = rho .* log(rho ./ rhoHat) + (1 - rho) .* log((1 - rho) ./ (1 - rhoHat));

end

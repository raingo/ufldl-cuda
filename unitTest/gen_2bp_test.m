function gen_2bp_test()

local_settings;
addpath(MATLAB_IMPL_DIR);

sparsity = 0.035;
beta = 5;
lambda = 3e-3;         % weight decay parameter

% nSamples = 100000;
nSamples = 100;
dInput = 192;
dHidden = 400;

dOutput = dInput;

genFunc = @rand;
d_type = 'double';

input = genFunc(dInput, nSamples, d_type);
W1 = genFunc(dHidden, dInput, d_type);
W2 = genFunc(dOutput, dHidden, d_type);
a2 = genFunc(dHidden, nSamples, d_type);
delta3 = genFunc(dOutput, nSamples, d_type);
rho = genFunc(dHidden, 1, d_type);

% n_iter = 500;
n_iter = 1;

tic
for i = 1:n_iter
[pGradW1, pGradW2, pGradb1, pGradb2] = twoLayerBP(input, delta3, W1, ...
    W2, a2, rho, beta, sparsity, lambda);
i
end
toc

test_name = 'test_2bp';
allvars = {'pGradW1', 'pGradW2', 'pGradb1', 'pGradb2', 'input', ...
    'delta3', 'W1', 'W2', 'a2', 'rho'};

save_test_mat;

end

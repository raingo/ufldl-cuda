function gen_2cost_test()

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

data = genFunc(dInput, nSamples, d_type);
W1 = genFunc(dHidden, dInput, d_type);
W2 = genFunc(dOutput, dHidden, d_type);
a2 = genFunc(dHidden, nSamples, d_type);
a3 = genFunc(dOutput, nSamples, d_type); % a3: dOutput * nSamples

% n_iter = 500;
n_iter = 1;

tic
for i = 1:n_iter
[cost, delta, rho] = twoLayerCost(data, a2, a3, lambda, W1, W2, ...
    sparsity, beta);
i
end
toc

test_name = 'test_2cost';
allvars = {'cost', 'delta', 'rho', 'data', 'a2', ...
    'a3', 'W1', 'W2'};

save_test_mat;

end

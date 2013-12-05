function gen_2grad_test()

    local_settings;
addpath(MATLAB_IMPL_DIR);

sparsity = 0.035;
beta = 5;
lambda = 3e-3;         % weight decay parameter

% nSamples = 100000;
nSamples = 1000;
dInput = 192;
dHidden = 400;

dOutput = dInput;

genFunc = @rand;
d_type = 'double';

theta = initializeParameters(dHidden, dInput, d_type);
data = genFunc(dInput, nSamples, d_type);

[cost,grad] = sparseAutoencoderLinearCost(theta, dInput, dHidden, ...
                                             lambda, sparsity, beta, data);

test_name = 'test_2grad';
allvars = {'cost', 'grad', 'theta', 'data'};

save_test_mat;

end

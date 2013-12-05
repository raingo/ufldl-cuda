function gen_2opt_test()

    local_settings;
addpath(MATLAB_IMPL_DIR);

sparsity = 0.035;
beta = 5;
lambda = 3e-3;         % weight decay parameter

nSamples = 100000;
% nSamples = 100;
dInput = 192;
dHidden = 400;

dOutput = dInput;

genFunc = @rand;
theta = initializeParameters(dHidden, dInput, 'single');
% data = genFunc(dInput, nSamples, 'single');

test_name = 'test_2opt_f';
allvars = {'theta'};

save_test_mat;

end

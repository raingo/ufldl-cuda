function gen_2ff_test()

    local_settings;
addpath(MATLAB_IMPL_DIR);

% nSamples = 100000;
nSamples = 100000;
dInput = 192;
dHidden = 400;

% nSamples = 10;
% dInput = 4;
% dHidden = 5;


dOutput = dInput;

genFunc = @randn;
d_type = 'double';
input = genFunc(dInput, nSamples, d_type);
W1 = genFunc(dHidden, dInput, d_type);
W2 = genFunc(dOutput, dHidden, d_type);
b1 = genFunc(dHidden, 1, d_type);
b2 = genFunc(dOutput, 1, d_type);

% n_iter = 500;
n_iter = 1;

tic
for i = 1:n_iter
    [a2, a3] = twoLayerFF(input, W1, W2, b1, b2);
    i
end
toc

test_name = 'test_2ff';

allvars = {'a2', 'a3', 'input', 'W1', 'W2', 'b1', 'b2'};

save_test_mat;

end

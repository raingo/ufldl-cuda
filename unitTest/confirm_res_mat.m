
local_settings;
save_dir = fullfile(base_dir, test_name);

for i = 1:length(allvars)
    var = allvars{i};
    load(fullfile(save_dir, [var, '.mat']));
    varTest = load(fullfile(save_dir, [var, 'res.mat']));
    varTest = varTest.(var);

    eval(sprintf('diff = varTest - %s; disp({''%s'', max(diff(:))});', var, var));
end

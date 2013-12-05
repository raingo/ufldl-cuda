
local_settings; % base_dir will be assigned

save_dir = fullfile(base_dir, test_name);

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
for i = 1:length(allvars)
    var = allvars{i};
%     eval(sprintf('%s = swap_idx_major(%s); %s = single(%s);', var, var, var, var));
    eval(sprintf('%s = single(%s);', var, var));
    save(fullfile(save_dir, [var, '.mat']), var);
end

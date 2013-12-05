function mat = swap_idx_major(mat)

sizeMat = size(mat);
mat = reshape(mat, sizeMat(2), sizeMat(1))';

end
function A = genSeqMat(row, col, ~)

A = 1:(row * col);
A = reshape(A, [row, col]);

end
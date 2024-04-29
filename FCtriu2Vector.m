function M_vector = FCtriu2Vector(M,flag)
    % Extract the upper triangular FC matrix (symmetric) into a vector by column 
    % If flag = 1, the diagonal elements of the matrix are not included.
    % If flag = 0, the diagonal elements of the matrix are included.
    true_mat = true(size(M,1),size(M,1));
    ind = triu(true_mat,flag);
    M_vector = M(ind);
end


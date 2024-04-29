function M = Vector2FCtriu(M_vector,flag)
    % Reshape a vector into the upper triangular FC matrix (symmetric) by column
    % If flag = 1, the diagonal elements of the matrix are not included.
    % If flag = 0, the diagonal elements of the matrix are included.
    % NB: The numnber of the elements of the column vector should be identical to the matrix to be reshaped.
    if flag
        n = (1+sqrt(1+8*length(M_vector)))/2;
        M = zeros(n);
        ind = triu(ones(n),flag);
        M(find(ind)) = M_vector;
    else
        n = (-1+sqrt(1+8*length(M_vector)))/2;
        M = zeros(n);
        ind = triu(ones(n),flag);
        M(find(ind)) = M_vector;
    end
end

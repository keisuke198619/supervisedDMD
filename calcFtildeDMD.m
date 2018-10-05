function [Ftilde, U_r, S_r, V_r] = calcFtildeDMD(X,Y,eps,rnk)
[U, S, V] = svd(X, 'econ');
eval   = diag(S); 
[~,IX] = sort(abs(eval),'descend');
if ~isempty(eps) && isempty(rnk)
    IX     = IX(abs(eval(IX))>eps);
elseif isempty(eps) && ~isempty(rnk)
    IX     = IX(1:rnk); 
end
U_r   = U(:,IX);
V_r   = V(:,IX);
S_r   = S(IX,IX);
Ftilde = U_r' * Y * V_r *diag(1./diag(S_r)); % low-rank dynamics
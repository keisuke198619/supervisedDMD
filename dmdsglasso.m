%% Sparse group lasso for supervised DMD
% Keisuke Fujii, 2018
% modifying the code of Mihailo R. Jovanovic, August 2012

function answer = dmdsglasso(P,q,s,gammaval,label,options)

% Data preprocessing
rho = options.rho;
Max_ADMM_Iter = options.maxiter;
eps_abs = options.eps_abs;
eps_rel = options.eps_rel;

% Number of optimization variables
n = length(q);
% Identity matrix
I = eye(n);
% label group
ulabel = unique(label) ;
gr = length(ulabel); 
 
% Allocate memory for gamma-dependent output variables
answer.gamma = gammaval;
answer.Nz    = zeros(1,size(gammaval,1)); % number of non-zero amplitudes 
answer.Jsp   = zeros(1,size(gammaval,1)); % square of Frobenius norm (before polishing)
answer.Jpol  = zeros(1,size(gammaval,1)); % square of Frobenius norm (after polishing)
answer.Ploss = zeros(1,size(gammaval,1)); % optimal performance loss (after polishing)
answer.xsp   = zeros(n,size(gammaval,1)); % vector of amplitudes (before polishing)
answer.xpol  = zeros(n,size(gammaval,1)); % vector of amplitudes (after polishing)

% Cholesky factorization of matrix P + (rho/2)*I
Prho = (P + (rho/2)*I);
Plow = chol(Prho,'lower');
Plow_star = Plow';

for i = 1:size(gammaval,1)
    
    gamma = gammaval(i,1);
    gamma2 = gammaval(i,2);
    % Initial conditions
    y = zeros(n,1); % Lagrange multiplier
    z = zeros(n,1); % copy of x
    
    % Use ADMM to solve the gamma-parameterized problem  
    for ADMMstep = 1: Max_ADMM_Iter
          
        % x-minimization step (alpha)
        u = z - (1/rho)*y;
        xnew = Plow_star\(Plow\(q + (rho/2)*u)); 
        
        % z-minimization step (beta)
        a = (gamma/rho)*ones(n,1);
        v = xnew + (1/rho)*y;
        
        % Soft-thresholding of v
        znew_l1 = ( (1 - a ./ abs(v)) .* v ) .* (abs(v) > a); % L1
        if gamma2 ~= 0 && ~isempty(label)% Group(L1/L2)
            a2 = (gamma2/rho)*ones(n,1);
            for g = 1:gr
                int = find(label==ulabel(g)) ;
                if sum(abs(znew_l1(int))) < 1e-6
                    znew(int,1) = 0 ;
                elseif norm(znew_l1(int)) <= a2(int(1))
                    znew(int,1) = 0 ;
                else
                    znew(int,1) = (1 - a2(int) ./ norm(znew_l1(int))) .* znew_l1(int);
                end
            end
        else ; znew = znew_l1 ;
        end
        % Primal and dual residuals
        res_prim = norm(xnew - znew);
        res_dual = rho * norm(znew - z);
        
        % Lagrange multiplier update step
        y = y + rho*(xnew - znew);
        
        % Stopping criteria
        eps_prim = sqrt(n) * eps_abs + eps_rel * max([norm(xnew),norm(znew)]);
        eps_dual = sqrt(n) * eps_abs + eps_rel * norm(y);
        
        if (res_prim < eps_prim) && (res_dual < eps_dual)
            break;
        else
            z = znew;
        end
        Jsps{i}(ADMMstep,1) = sqrt(real(z'*P*z) - 2*real(q'*z) + s);
    end             
    
    % Record output data
    answer.xsp(:,i) = z; % vector of amplitudes
    answer.Nz(i) = nnz(answer.xsp(:,i)); % number of non-zero amplitudes
    answer.Jsp(i) = sqrt(real(z'*P*z) - 2*real(q'*z) + s); % Frobenius norm (before polishing)
    answer.Jsp2(i) = sqrt(answer.Jsp(i)/s);     
    if 1 % Polishing of the nonzero amplitudes
        % Form the constraint matrix E for E^T x = 0
        ind_zero = find( abs(z) < 1.e-12); % find indices of zero elements of z
        m = length(ind_zero); % number of zero elements
        E = I(:,ind_zero);
        E = sparse(E);
        
        % Form KKT system for the optimality conditions
        KKT = [P, E; E', zeros(m,m)];
        rhs = [q; zeros(m,1)];
        
        % Solve KKT system
        sol = KKT \ rhs;
        
        % Vector of polished (optimal) amplitudes
        xpol = sol(1:n);
        
        % Record output data
        answer.xpol(:,i) = xpol;
        % Polished (optimal) least-squares residual
        answer.Jpol(i) = sqrt(real(xpol'*P*xpol) - 2*real(q'*xpol) + s);
        answer.Jpol2(i) = sqrt(answer.Jpol(i)/s);
        % Polished (optimal) performance loss
        answer.Ploss(i) = 100*sqrt(answer.Jpol(i)/s);
    end
    
    %i
    answer.iter = ADMMstep ;
end     

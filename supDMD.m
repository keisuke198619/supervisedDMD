% supDMD.m
% perform supervised DMD, group DMD, exact DMD, and sparsity-promoting DMD (spDMD)
% Keisuke Fujii

function [Lam,Lamg,Lamt,Lamsp,Phi,Phig,Phit,Phisp,obj,obj0] = supDMD(x_t, params)
% W:     DMD eigenvector
% Lam:   DMD eigenvalue
% Phi:   DMD mode
% alpha: amplitude of DMD mode

for i = 1:size(params.regs,1)
    [Lam{i},Lamg{i},Lamt{i},Lamsp{i},Phi{i},Phig{i},Phit{i},Phisp{i},obj{i},obj0{i}] = ...
        estimate_with_fixed_reg(x_t, params.regs(i,:), params);
end
end

function [Lamnew,Lamnew_g,Lam_t,Lam_sp,Phinew,Phinew_g,Phi_t,Phi_sp,obj,obj0] =...
    estimate_with_fixed_reg(x_t, reg, params)
% Set options for sparse cording
options = struct('rho',1,'maxiter',10000,'eps_abs',params.eps_admm_abs,'eps_rel',params.eps_admm_rel);

dt = params.dt ;
% solve each sequence (exact DMD)
for k = 1:params.K
    Xt{k} = x_t{k}(:,1:end-1) ;
    Yt{k} = x_t{k}(:,2:end) ;
    [Ftilde_t{k}, U_t{k}, S_t{k}, V_t{k}] = calcFtildeDMD(Xt{k},Yt{k},params.eps_dmd,params.rank_dmd);
    rr(k,1) = rank(Ftilde_t{k}) ;
    r_t(k,1) = size(Ftilde_t{k},1); % set the number of modes
    [W_t{k}, Lam_t{k}, Z_t{k}] = eig(Ftilde_t{k});
    Phi_t{k} = Yt{k} * V_t{k} * diag(1./diag(S_t{k})) * W_t{k} * diag(1./diag(Lam_t{k})) ;
    if params.spDMD % spDMD
        tau = size(Xt{k},2) ;
        V1_t = zeros(size(Ftilde_t{k},1),tau);
        for i = 1:tau
            V1_t(:,i) = diag(Lam_t{k}).^(i-1);
        end
        
        L = W_t{k}; 
        R = V1_t;
        G = U_t{k}'*Yt{k}; 
        % Form matrix P, vector q, and scalar s
        % J = x'*P*x - q'*x - x'*q + s % x - optimization variable
        P = (L'*L).*conj(R*R');
        q = conj(diag(R*G'*L));
        s = trace(G'*G);
        % Sparsity-promoting algorithm
        ansDMDsp_t = dmdsp(P,q,s,reg(1),options); 
        alpha_t0 = ansDMDsp_t.xsp(:,end);
   
        ind = abs(alpha_t0)>params.eps_sp;
        sparsity = sum(ind);
        if sparsity == 0
            error(['all zero when rho= ',num2str(reg)])
        end
        
        % solve DMD again
        Wnew = G*pinv(diag(alpha_t0(ind))*V1_t(ind,:));
        Lam_sp{k} = Lam_t{k}(ind,ind);
        Phi_sp{k} = Yt{k} * V_t{k} * diag(1./diag(S_t{k}))...
            * Wnew * diag(1./diag(Lam_sp{k})) ;
        
        % reconstruction error
        Wnew0 = G*pinv(diag(alpha_t0)*V1_t);
        tmp0 = U_t{k}'*Yt{k} ;
        tmp1 = Wnew0*Lam_t{k}*pinv(Wnew0)* U_t{k}'* Xt{k}; % W_t{k}
        tmp = tmp0 - tmp1 ;
        objk0(k) = sqrt(trace(tmp'*tmp)) ; % + reg(1)*sum(abs(alpha_t0)) ;
    else Lam_sp{k} = [] ; Phi_sp{k} = [] ;
    end

end
if params.spDMD
    obj0{5} = sum(objk0)  ;
end

[d1,tau] = size(Xt{1}) ;

% 1. jointly solve first DMD (for sup DMD and group DMD) ------------------------------
X = [] ; Y = [];
for n = 1:params.K
    X = cat(1,X,Xt{n}) ;
    Y = cat(1,Y,Yt{n}) ;
end
[Ftilde, U, S, V] = calcFtildeDMD(X,Y,params.eps_dmd,params.rank_dmd);
[W, Lam, Z] = eig(Ftilde);
% Form Vandermonde matrix
V1 = zeros(size(Ftilde,1),tau);
for i = 1:tau
    V1(:,i) = diag(Lam).^(i); 
end

% reconstruction error in first DMD
tmp0 = U'* Y ;
tmp = U'* Y- Ftilde* U'* X ;
obj{1} = sqrt(trace(tmp'*tmp))/sqrt(trace(tmp0'*tmp0));
obj0{1} = sqrt(trace(tmp'*tmp));

% 2. sparse coding (given fix W, solve alpha) ------------------------------
ind_label(:,2) = cumsum(params.nTask) ;
ind_label(2:end,1) = ind_label(1:end-1,2)+1 ;
ind_label(1,1) = 1 ;
for g = 1:params.nGroup
    Xg{g} = [] ; Yg{g} = [];
    for n = 1:params.nTask(g)
        Xg{g} = cat(1,Xg{g},Xt{ind_label(g,1)-1+n}) ;
        Yg{g} = cat(1,Yg{g},Yt{ind_label(g,1)-1+n}) ;
    end
end

% J = ||U'Y1 - F*W*diag(alpha)*V1 ||^2
%   = ||G    - L  *diag(alpha)*R ||^2
Lb = [] ; Rb = [] ;Gb = [] ; 
L = W;
tmpU = U ; 

for g = 1:params.nGroup % shared among all label
    Ug{g} = tmpU(1:size(Yg{g},1),:) ;
    tmpU(1:size(Yg{g},1),:) = [];
    Gg{g} = Ug{g}'*Yg{g} ;
    Gb = cat(1,Gb,Gg{g});
    Lb = blkdiag(Lb,L); %
    Rb = cat(1,Rb,V1);
end

d = size(Ftilde,1) ;
ng = params.nGroup ;
label =  repmat([1:d]',ng,1) ;

% Form big matrix P, vector q, and scalar s
P = (Lb'*Lb).*conj(Rb*Rb');
q = conj(diag(Rb*Gb'*Lb));
s = trace(Gb'*Gb);

% sparse group lasso
ansDMDsp = dmdsglasso(P,q,s,reg,label,options);
alpha1 = ansDMDsp.xpol(:,end) ; % xsp / xpol

% reconstruction error in sparse coding
obj0{2} = real(ansDMDsp.Jpol(:,end)); %Jsp/Jpol ;
obj{2} = real(ansDMDsp.Jpol2(:,end)); %Jsp2/Jpol2 ;
spiter = ansDMDsp.iter ;%

if params.nGroup > 1 % group DMD
    label = 1:d;
    G = U'*Y ;
    P = (L'*L).*conj(V1*V1');
    q = conj(diag(V1*G'*L));
    s = trace(G'*G);
    ansDMDsp = dmdsglasso(P,q,s,reg,label,options);
    alpha_g0 = ansDMDsp.xpol(:,end) ; % xsp / xpol
    obj{5} = real(ansDMDsp.Jpol(:,end)); %Jsp/Jpol ;
end

% 3. solve DMD again (given fix alpha, solve each F) -----------------------------
alpha2 = alpha1 ; 
% check sparsity
for g = 1:params.nGroup
    alpha_g{g} = alpha2(1:size(Ftilde,1));
    alpha2(1:size(Ftilde,1)) = [] ;
    ind_g{g} = abs(alpha_g{g})>params.eps_sp;
    sparsity = sum(ind_g{g});
    if sparsity == 0
        error(['all zero when rho= ',num2str(reg),' group = ',num2str(g)])
    end
end

for g = 1:params.nGroup

    % solve DMD again
    Wnew = Gg{g}(ind_g{g},:)*pinv(diag(alpha_g{g}(ind_g{g}))*V1(ind_g{g},:));
    wnew = log(diag(Lam(ind_g{g},ind_g{g})))/dt/(2*pi);
    % truncate
    Phi = Yg{g} * V(:,ind_g{g}) * diag(1./diag(S(ind_g{g},ind_g{g}))) * Wnew ...
        * diag(1./diag(Lam(ind_g{g},ind_g{g}))) ;
    % seperate into tasks
    for nn = 1:params.nTask(g)
        n = ind_label(g,1) -1 + nn ;
        Phinew{n} = Phi((nn-1)*d1+1:nn*d1,:) ;
        Lamnew{n} = Lam(ind_g{g},ind_g{g}) ;
    end
    alphanew{g} = alpha_g{g}(ind_g{g}) ;
    % reconstruction error in sup DMD
    Wnew0 = Gg{g}*pinv(diag(alpha_g{g})*V1);
    tmp0 = Ug{g}'* Yg{g} ;
    tmp1 = Wnew0*Lam*pinv(Wnew0)* Ug{g}'* Xg{g};
    tmp = tmp0 - tmp1 ;
 
    objk2(g) = sqrt(trace(tmp'*tmp)) ;%+ reg(2)*norm(alpha_g{g}) ;
end

% reconstruction error in sup DMD
obj0{3} = sum(objk2) ;%+ reg(1)*sum(abs(alpha)) ; 

if params.nGroup > 1 % group DMD
    % solve DMD
    % [~,IX] = sort(abs(alpha_g0),'descend');
    IX = find(abs(alpha_g0)>params.eps_sp);
    Wnew = G(IX,:)*pinv(diag(alpha_g0(IX))*V1(IX,:));
    % truncate
    Phi = Y * V(:,IX) * diag(1./diag(S(IX,IX))) * Wnew * diag(1./diag(Lam(IX,IX))) ;
    % seperate into tasks
    for n = 1:params.K
        Phinew_g{n} = Phi((n-1)*d1+1:n*d1,:) ;
        Lamnew_g{n} = Lam(IX,IX) ;
    end
    % reconstruction error in group DMD
    Wnew0 = G*pinv(diag(alpha_g0)*V1);
    tmp0 = U'* Y ;
    tmp1 = Wnew0*Lam*pinv(Wnew0)* U'* X;
    tmp = tmp0 - tmp1 ;
    obj0{4} = sqrt(trace(tmp'*tmp)) ;%+ reg(1)*sum(abs(alpha_g0));
else ; Phinew_g = [] ;Lamnew_g = [] ; 
end

% display information
disp(['rho1=',num2str(reg(1)),' rho2=',num2str(reg(2)),' t=',num2str(spiter),...
    ' nalp=',num2str(mean(sparsity)),...
    ' Jdmd1:',num2str(obj0{1},'%10.3e'),...
    ' Jsp(+reg):',num2str(obj0{2},'%10.3e'),...
    ' Jdmd2:',num2str(obj0{3},'%10.3e')]);
end

% demo_supDMD.m
% create Fig.1a in submitting paper
% Keisuke Fujii

clear ; close all ;
dbstop if error

% Generate Data
params.lt = 200; % time length
params.lx = 10;  % data dimension
params.omega0 = 2.3 ; % common frequency
params.omegas = [2.8,3.3,3.8]; % label-specific frequencies  
params.nTask = [5 5 5]; % No. of sequences for each label
params.nGroup = 3 ; % No. of labels
params.K = sum(params.nTask); % No. of all sequences
params.noise_level = 0.5; % noise level
rng(0, 'twister');
[x_t, params,time] = generate_data(params); 

% parameters
params.regs = [6 40]; % Regularization parameter  
params.eps_dmd = 1e-3 ; % tolerance for DMD 
params.rank_dmd = [] ; % if you specify the rank  
params.eps_sp = 1e-3 ; % tolerance for sparse cording 
params.eps_admm_abs = 1e-6 ; % tolerance for ADMM 
params.eps_admm_rel = 1e-4 ; % see the papar of Jovanovic et al.(2014) about sparsity-promoting DMD
params.spDMD = 1 ; % if you compute sparsity-promoting DMD

% supervised, group, exact, sparsity-promoting DMD 
[Lamc, Lamg, Lamt, Lamsp, Phic, Phig, Phit, Phisp,obs,obs0] = supDMD(x_t, params);

% L1 + label
params.regs = [params.regs(1) 0 ];
[LamL1, ~, ~, ~, PhiL1] = supDMD(x_t, params);

% create figure and tables
figure(1)
Om = 1.95:0.05:3.95 ;
str = {'sp-DMD','L1+label','supervise','group'} ;
FS = 10 ;
Lam0 = exp(-params.omega0*1i*params.dt);
Lams = exp(-params.omegas*1i*params.dt);
for ty = 1:4
    Tab{ty} = zeros(params.K,length(Om)); % frequency-sequence (grid: Figure 1)
    for k = 1:params.K
        if ty == 1
            lambdac = diag(Lamsp{1}{k});
            Phi = abs(Phisp{1}{k});
%             lambdac = diag(Lamt{1}{k}); % similar 
%             Phi = abs(Phit{1}{k});
        elseif ty == 2
            lambdac = diag(LamL1{1}{k});
            Phi = abs(PhiL1{1}{k});
        elseif ty == 3
            lambdac = diag(Lamc{1}{k});
            Phi = abs(Phic{1}{k});
        elseif ty == 4
            lambdac = diag(Lamg{1}{k});
            Phi = abs(Phig{1}{k});
        end
        om = abs(imag(log(lambdac)/params.dt));
        % for figure
        Tab2{ty}(k,1:length(om)) = om ; % frequency-sequence (obtained)
        for r = 1:length(Om)-1
            ind2 = and(om>Om(r),om<Om(r+1));
            if sum(ind2)~= 0
                Tab{ty}(k,r) = mean(mean(abs(Phi(:,ind2)))) ;
            end
        end
        % Table of estimation error (lambda and omega)
        Tab3(ty,1) = abs(min(Lam0-lambdac)); 
        Tab3(ty,3) = min(abs(params.omega0-om)); 
        if k <= 5
            Tab3(ty,2) = min(abs(Lams(1)-lambdac)); 
            Tab3(ty,4) = min(abs(params.omegas(1)-om)); 
        elseif k <= 10
            Tab3(ty,2) = min(abs(Lams(2)-lambdac)); 
            Tab3(ty,4) = min(abs(params.omegas(2)-om)); 
        elseif k <= 15
            Tab3(ty,2) = min(abs(Lams(3)-lambdac)); 
            Tab3(ty,4) = min(abs(params.omegas(3)-om)); 
        end
    end
    % plot
    subplot(5,1,ty+1)
    imagesc(Tab{ty},[0 0.04])
    colormap gray
    
    if ty == 4
        xlabel('frequency(Hz)', 'FontName','Arial','FontSize',FS)
        set(gca,'xtick',1:10:41,'xticklabel',2:0.5:4, 'FontName','Arial','FontSize',FS)
    else set(gca,'xtick',[],'xticklabel',[], 'FontName','Arial','FontSize',FS);
    end
    ylabel(str{ty})
end

% True frequency
TrueTab = zeros(params.K,length(Om)) ;
TrueTab(:,7) = 1 ; TrueTab(1:5,17) = 1 ;
TrueTab(6:10,27) = 1 ; TrueTab(11:15,38) = 1 ;
subplot(5,1,1)
imagesc(TrueTab,[0 0.5])
ylabel('True', 'FontName','Arial','FontSize',FS)  
set(gca,'xtick',[],'xticklabel',[], 'FontName','Arial','FontSize',FS);



function [x_t, params,t] = generate_data(params)

w0 = params.omega0 ;
ws = params.omegas ;
lt = params.lt ;
lx = params.lx ;
nTask = params.nTask;
nw = length(ws) ;
nall = sum(params.nTask) ;
%% Define time and space discretizations
t = linspace(0,4*pi,lt);
dt = t(2) - t(1);
nl = params.noise_level ;
xbound = 5 ;
for n = 1:nall
    rs(n) = randn() ; % within label
    tt{n} = t ;
    x0 = linspace(-xbound,xbound,lx);
    [Xgrid0{n},T{n}] = meshgrid(x0,tt{n});
    [Xgrid{n},T{n}] = meshgrid(x0,tt{n});
end
%% Create two spatio-temporal patterns
% common pattern
for n = 1:nall
    f0{n} = sech(Xgrid0{n}+3) .* (1*exp(1j*w0*T{n}));
end
% label-specific pattern
n = 1 ;
for w = 1:nw
    for nn = 1:nTask(w)
        fs{n} = (sech(Xgrid{n}).*tanh(Xgrid{n})).*(1*exp(1j*ws(w)*T{n}))+nl*rand(lt,lx);
        x_t{n} = (f0{n} + fs{n})' ;%
        n = n + 1 ;
    end
end
params.dt = dt;
end


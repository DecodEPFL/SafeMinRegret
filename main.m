clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
rng(1234);             % Set random seed for reproducibility
%% Definition of the underlying discrete-time LTI system
sys.rho = 0.55; % Spectral radius
sys.A = sys.rho*[0.7 0.2 0; 0.3 0.7 -0.1; 0 -0.2 0.8];
sys.B = [1 0.2; 2 0.3; 1.5 0.5];

sys.n = size(sys.A, 1);   % Order of the system: state dimension
sys.m = size(sys.B, 2);   % Number of input channels
sys.x0 = zeros(sys.n, 1); % Initial condition

sys.Hu = [eye(sys.m); -eye(sys.m)]; % Polytopic constraints: Hu * u <= hu
sys.hu = 1*ones(size(sys.Hu, 1), 1);

sys.Hx = [eye(sys.n); -eye(sys.n)]; % Polytopic constraints: Hx * x <= hx
sys.hx = 2*ones(size(sys.Hx, 1), 1);

sys.Hw = [eye(sys.n); -eye(sys.n)]; % Polytopic disturbance set: Hw * w <= hw 
sys.hw = 1*ones(size(sys.Hw, 1), 1);
%% Definition of the parameters of the optimization problem
opt.Q = eye(sys.n); % State weight matrix
opt.R = eye(sys.m); % Input weight matrix

opt.T = 12; % Control horizon (even and multiple of 3)
%% Definition of the stacked system dynamics over the control horizon
sls.A = kron(eye(opt.T), sys.A);
sls.B = kron(eye(opt.T), sys.B);

sls.I = eye(sys.n*opt.T); % Identity matrix and block-downshift operator
sls.Z = [zeros(sys.n, sys.n*(opt.T-1)) zeros(sys.n, sys.n); eye(sys.n*(opt.T-1)) zeros(sys.n*(opt.T-1), sys.n)];

% Polytopic disturbance description and safety constraints
sls.Hu = kron(eye(opt.T), sys.Hu);
sls.hu = kron(ones(opt.T, 1), sys.hu);

sls.Hx = kron(eye(opt.T), sys.Hx);
sls.hx = kron(ones(opt.T, 1), sys.hx);

sls.H = blkdiag(sls.Hu, sls.Hx);
sls.h = [sls.hu; sls.hx];

sls.Hw = kron(eye(opt.T), sys.Hw);
sls.hw = kron(ones(opt.T, 1), sys.hw);
%% Computation of the optimal noncausal unconstrained controller
% The optimal dynamic sequence of control actions is unique.
% However, the optimal H2 and Hinf costs incurred by the clairvoyant
% controller are different!
[Phi_nc_unc.x, Phi_nc_unc.u, obj_nc.unc_h2, obj_nc.unc_hinf] = noncausal_unconstrained_benchmark(sys, sls, opt);
%% Computation of the optimal noncausal constrained H2 and Hinf controller
[Phi_nc_con_h2.x,   Phi_nc_con_h2.u,   obj_nc.con_h2]   = noncausal_constrained_benchmark(sys, sls, opt, 'H2');
[Phi_nc_con_hinf.x, Phi_nc_con_hinf.u, obj_nc.con_hinf] = noncausal_constrained_benchmark(sys, sls, opt, 'Hinf');
%% Computation of the optimal causal unconstrained H2 and Hinf controller
[Phi_c_unc_h2.x,   Phi_c_unc_h2.u,   obj_c.unc_h2]   = causal_unconstrained(sys, sls, opt, 'H2');
[Phi_c_unc_hinf.x, Phi_c_unc_hinf.u, obj_c.unc_hinf] = causal_unconstrained(sys, sls, opt, 'Hinf');
%% Computation of the optimal causal constrained H2 and Hinf controller
[Phi_c_con_h2.x,   Phi_c_con_h2.u,   obj_c.con_h2]   = causal_constrained(sys, sls, opt, 'H2');
[Phi_c_con_hinf.x, Phi_c_con_hinf.u, obj_c.con_hinf] = causal_constrained(sys, sls, opt, 'Hinf');
%% Computation of the regret-optimal causal controller
[Phi_reg_unc_nc_unc.x, Phi_reg_unc_nc_unc.u, obj_reg.unc_nc_unc] = regret_unconstrained(sys, sls, opt, Phi_nc_unc);
[Phi_reg_con_nc_unc.x, Phi_reg_con_nc_unc.u, obj_reg.con_nc_unc] = regret_constrained(sys, sls, opt, Phi_nc_unc);
[Phi_reg_con_nc_con_h2.x,   Phi_reg_con_nc_con_h2.u,   obj_reg.con_nc_con_h2]   = regret_constrained(sys, sls, opt, Phi_nc_con_h2);
[Phi_reg_con_nc_con_hinf.x, Phi_reg_con_nc_con_hinf.u, obj_reg.con_nc_con_hinf] = regret_constrained(sys, sls, opt, Phi_nc_con_hinf);
%% Numerical experiments: compare time-averaged incurred LQR cost
clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
load('data_T30_rho0p55_hu1_hx2_hw1.mat'); % Example with a stable system
%load('data_T30_rho1p05_hu10_hx10_hw1');   % Example with an unstable system

profiles = 8;
iter = 1 + 999*[ones(3, 1); zeros(5, 1)]; % Maximum number of iterations per profile
for i = 1:profiles
    for j = 1:iter(i)
        % Sample a disturbance realization
        if i == 1     % N(0, 1)
            w = [sys.x0; randn(sys.n*opt.T - sys.n, 1)];
            if j == iter(i)
                fprintf('\nGaussian: N(0,1)\n\n')
            end
        elseif i == 2 % U(0.5, 1)
            w = [sys.x0; 0.5 + rand(sys.n*opt.T - sys.n, 1)*0.5];
            if j == iter(i)
                fprintf('\nUniform: U(0.5, 1)\n\n')
            end
        elseif i == 3 % U(0, 1)
            w = [sys.x0; rand(sys.n*opt.T - sys.n, 1)];
            if j == iter(i)
                fprintf('\nUniform: U(0, 1)\n\n')
            end
        elseif i == 4 % CONSTANT
            w = [sys.x0; ones(sys.n*opt.T - sys.n, 1)];
            fprintf('\nConstant\n\n')
        elseif i == 5 % SIN
            w = [sys.x0; sin(0.1*(1:opt.T-1))'; sin(0.1*(1:opt.T-1))'; sin(0.1*(1:opt.T-1))'];
            fprintf('\nSinusoidal wave\n\n')
        elseif i == 6 % SAWTOOTH
            w = [sys.x0; sawtooth(0.1*(1:opt.T-1), 1)'; sawtooth(0.1*(1:opt.T-1), 1)'; sawtooth(0.1*(1:opt.T-1), 1)']; 
            fprintf('\nSawtooth wave\n\n')
        elseif i == 7 % STEP
            w = [sys.x0; zeros(sys.n*opt.T/2 - sys.n, 1); ones(sys.n*opt.T/2, 1)];
            fprintf('\nStep function\n\n')
        else          % STAIRS
            w = [sys.x0; -ones(sys.n*opt.T/3 - sys.n, 1); zeros(sys.n*opt.T/3, 1); ones(sys.n*opt.T/3, 1)]; 
            fprintf('\nStairs function\n\n')
        end
        w = w(:);
        
        % Simulate the system with the optimal noncausal unconstrained controller
        [~, ~, nc_unc.ta_cum_costs(j, :)] = evaluate_policy(sys, opt, Phi_nc_unc, w);
        % Simulate the system with the optimal causal constrained H2 and Hinf controller
        [~, ~, c_con_h2.ta_cum_costs(j, :)]   = evaluate_policy(sys, opt, Phi_c_con_h2, w); 
        [~, ~, c_con_hinf.ta_cum_costs(j, :)] = evaluate_policy(sys, opt, Phi_c_con_hinf, w);
        % Simulate the system with the regret-optimal causal controller
        [~, ~, reg_con_nc_unc.ta_cum_costs(j, :)] = evaluate_policy(sys, opt, Phi_reg_con_nc_unc, w);
    end
    
    % Compute the mean cumulative cost incurred by the optimal noncausal unconstrained controller
    nc_unc.avg_cost = mean(nc_unc.ta_cum_costs(:, end));
    % Compute the mean cumulative cost incurred by the optimal causal constrained H2 and Hinf controller
    c_con_h2.avg_cost   = mean(c_con_h2.ta_cum_costs(:, end));
    c_con_hinf.avg_cost = mean(c_con_hinf.ta_cum_costs(:, end));
    % Compute the mean cumulative cost incurred by the regret-optimal causal controller
    reg_con_nc_unc.avg_cost = mean(reg_con_nc_unc.ta_cum_costs(:, end));
    
    % Display the average incurred control cost 
    fprintf('Constrained H2: %f\n', c_con_h2.avg_cost)
    fprintf('Constrained Hinf: %f\n', c_con_hinf.avg_cost)
    fprintf('Constrained regret: %f\n\n', reg_con_nc_unc.avg_cost)
    % Display the average control cost increase relative to the best policy
    if reg_con_nc_unc.avg_cost <  c_con_h2.avg_cost && reg_con_nc_unc.avg_cost < c_con_hinf.avg_cost
        fprintf('SH2/SR: %f   ' , (c_con_h2.avg_cost   - reg_con_nc_unc.avg_cost) / reg_con_nc_unc.avg_cost*100)
        fprintf('SHinf/SR: %f\n', (c_con_hinf.avg_cost - reg_con_nc_unc.avg_cost) / reg_con_nc_unc.avg_cost*100)
    elseif c_con_h2.avg_cost < reg_con_nc_unc.avg_cost && c_con_h2.avg_cost < c_con_hinf.avg_cost
        fprintf('SR/SH2: %f   '  , (reg_con_nc_unc.avg_cost - c_con_h2.avg_cost) / c_con_h2.avg_cost*100)
        fprintf('SHinf/SH2: %f\n', (c_con_hinf.avg_cost     - c_con_h2.avg_cost) / c_con_h2.avg_cost*100)
    else
        fprintf('SR/SHinf: %f   ', (reg_con_nc_unc.avg_cost - c_con_hinf.avg_cost) / c_con_hinf.avg_cost*100)
        fprintf('SH2/SHinf: %f\n', (c_con_h2.avg_cost       - c_con_hinf.avg_cost) / c_con_hinf.avg_cost*100)
    end
    
    clear nc_unc c_con_h2 c_con_hinf reg_con_nc_unc;
end
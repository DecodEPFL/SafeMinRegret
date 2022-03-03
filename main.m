clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
rng(1234);             % Set random seed for reproducibility
%% Definition of the underlying discrete-time LTI system
sys.rho = 0.7; % Spectral radius
sys.A = sys.rho*[0.7 0.2 0; 0.3 0.7 -0.1; 0 -0.2 0.8];
sys.B = [1 0.2; 2 0.3; 1.5 0.5];

sys.n = size(sys.A, 1);   % Order of the system: state dimension
sys.m = size(sys.B, 2);   % Number of input channels
sys.x0 = zeros(sys.n, 1); % Initial condition

sys.Hu = [eye(sys.m); -eye(sys.m)]; % Polytopic constraints: Hu * u <= hu
sys.hu = 2*ones(size(sys.Hu, 1), 1);

sys.Hx = [eye(sys.n); -eye(sys.n)]; % Polytopic constraints: Hx * x <= hx
sys.hx = 3*ones(size(sys.Hx, 1), 1);

sys.Hw = [eye(sys.n); -eye(sys.n)]; % Polytopic disturbance set: Hw * w <= hw 
sys.hw = 1*ones(size(sys.Hw, 1), 1);
%% Definition of the parameters of the optimization problem
opt.Qt = eye(sys.n); % Stage cost: state weight matrix
opt.Rt = eye(sys.m); % Stage cost: input weight matrix

opt.T = 10; % Control horizon

opt.Q = kron(eye(opt.T), opt.Qt); % State cost matrix
opt.R = kron(eye(opt.T), opt.Rt); % Input cost matrix
opt.C = blkdiag(opt.Q, opt.R); % Cost matrix
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
[Phi_nc_unc.x, Phi_nc_unc.u, obj_nc.unc_h2, obj_nc.unc_hinf] = noncausal_unconstrained(sys, sls, opt);
%% Computation of the optimal noncausal constrained H2 and Hinf controller
[Phi_nc_con_h2.x,   Phi_nc_con_h2.u,   obj_nc.con_h2]   = noncausal_constrained(sys, sls, opt, 'H2');
[Phi_nc_con_hinf.x, Phi_nc_con_hinf.u, obj_nc.con_hinf] = noncausal_constrained(sys, sls, opt, 'Hinf');
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
%% Numerical experiments: comparison between time-averaged incurred control cost
clc; close all; clear;
addpath('./functions') % Add path to the folder with auxiliary functions
%load('data_T30_rho0p55_hu1_hx2_hw1.mat'); % Example with an open-loop stable system
%load('data_T30_rho0p7_hu2_hx3_hw1.mat');  % Example with an open-loop stable system
load('data_T30_rho1p05_hu10_hx10_hw1');    % Example with an open-loop unstable system

disturbance.profiles = ["Gaussian: N(0,1)" "Uniform: U(0.5, 1)" "Uniform: U(0, 1)" "Constant" "Sinusoidal wave" "Sawtooth wave" "Step function" "Stairs function" "Worst-case"];
disturbance.stochastic = [1000*ones(3, 1); ones(6, 1)]; % Maximum number of iterations per profile

for i = 1:size(disturbance.profiles, 2) % Iterate over all different disturbance profiles
    for j = 1:disturbance.stochastic(i) 
        % Sample a disturbance realization
        if i == 1     % Gaussian: N(0, 1)
            w = [sys.x0; randn(sys.n*(opt.T - 1), 1)];
        elseif i == 2 % Uniform: U(0.5, 1)
            w = [sys.x0; 0.5 + rand(sys.n*(opt.T - 1), 1)*0.5];
        elseif i == 3 % Uniform: U(0, 1)
            w = [sys.x0; rand(sys.n*(opt.T - 1), 1)];
        elseif i == 4 % Constant at 1
            w = [sys.x0; ones(sys.n*(opt.T - 1), 1)];
        elseif i == 5 % Sinusoidal wave
            w = [sys.x0; kron(ones(sys.n, 1), sin(0.1*(1:opt.T-1)'))];
        elseif i == 6 % Sawtooth wave
            w = [sys.x0; kron(ones(sys.n, 1), sawtooth(0.1*(1:opt.T-1), 1)')];
        elseif i == 7 % Step function
            w = [sys.x0; zeros(sys.n*(opt.T - 1 - floor(opt.T/2)), 1); ones(sys.n*floor(opt.T/2), 1)];
        elseif i == 8 % Stairs function taking values in the set {-1, 0, 1}
            w = [sys.x0; -ones(sys.n*(opt.T - 1 - 2*floor(opt.T/3)), 1); zeros(sys.n*floor(opt.T/3), 1); ones(sys.n*floor(opt.T/3), 1)]; 
        else          % Worst-case disturbance: adversarial selection for all three safe control laws 
            % Compute the matrix that defines the quadratic form for the
            % cost incurred by the H2, Hinfinity and regret-optimal controller
            c_con_hinf.cost_qf = [Phi_c_con_hinf.x; Phi_c_con_hinf.u]'*opt.C*[Phi_c_con_hinf.x; Phi_c_con_hinf.u];
            c_con_h2.cost_qf   = [Phi_c_con_h2.x;   Phi_c_con_h2.u]'  *opt.C*[Phi_c_con_h2.x;   Phi_c_con_h2.u];
            reg_con_nc_unc.cost_qf = [Phi_reg_con_nc_unc.x; Phi_reg_con_nc_unc.u]'*opt.C*[Phi_reg_con_nc_unc.x; Phi_reg_con_nc_unc.u];
            % Extract the direction of the eigenvector associated with the
            % largest eigenvalue while maintaining zero initial condition
            [c_con_hinf.evectors, c_con_hinf.evalues] = eig(c_con_hinf.cost_qf(sys.n+1:end, sys.n+1:end), 'vector');
            [c_con_h2.evectors,   c_con_h2.evalues]   = eig(c_con_h2.cost_qf(sys.n+1:end, sys.n+1:end), 'vector');
            [reg_con_nc_unc.evectors, reg_con_nc_unc.evalues] = eig(reg_con_nc_unc.cost_qf(sys.n+1:end, sys.n+1:end), 'vector');
            [~, c_con_hinf.index] = max(c_con_hinf.evalues);
            [~, c_con_h2.index]   = max(c_con_h2.evalues);
            [~, reg_con_nc_unc.index]  = max(reg_con_nc_unc.evalues);
            c_con_hinf.w = [sys.x0; c_con_hinf.evectors(:, c_con_hinf.index)]; 
            c_con_h2.w   = [sys.x0; c_con_h2.evectors(:, c_con_h2.index)]; 
            reg_con_nc_unc.w  = [sys.x0; reg_con_nc_unc.evectors(:, reg_con_nc_unc.index)]; 
        end
        if i ~= 9 % Always simulate the three considered safe control policies with the same disturbance sequence, except when dealing with the worst-case of each of them
            c_con_hinf.w = w/norm(w); c_con_h2.w = w/norm(w); reg_con_nc_unc.w = w/norm(w);
        end
        c_con_hinf.w = c_con_hinf.w(:); c_con_h2.w = c_con_h2.w(:); reg_con_nc_unc.w = reg_con_nc_unc.w(:); % Vectorize the sampled disturbance sequence
        
        % Simulate the closed-loop system with the optimal causal constrained H2 and Hinf controller
        c_con_h2.cum_costs(j)   = evaluate_policy(opt, Phi_c_con_h2, c_con_h2.w); 
        c_con_hinf.cum_costs(j) = evaluate_policy(opt, Phi_c_con_hinf, c_con_hinf.w);
        % Simulate the closed-loop system with the regret-optimal causal controller
        reg_con_nc_unc.cum_costs(j) = evaluate_policy(opt, Phi_reg_con_nc_unc, reg_con_nc_unc.w);
    
    end
    
    % Compute the mean cumulative cost incurred by the optimal causal constrained H2 and Hinf controller
    c_con_h2.avg_cost   = mean(c_con_h2.cum_costs);
    c_con_hinf.avg_cost = mean(c_con_hinf.cum_costs);
    % Compute the mean cumulative cost incurred by the regret-optimal causal controller
    reg_con_nc_unc.avg_cost = mean(reg_con_nc_unc.cum_costs);

    % Display the average incurred control costs 
    fprintf('%s\n\n', disturbance.profiles(i))
    fprintf('Constrained H2: %f\n', c_con_h2.avg_cost)
    fprintf('Constrained Hinf: %f\n', c_con_hinf.avg_cost)
    fprintf('Constrained regret: %f\n\n', reg_con_nc_unc.avg_cost)
    
    % Display the average control cost increase relative to the best policy
    if reg_con_nc_unc.avg_cost <  c_con_h2.avg_cost && reg_con_nc_unc.avg_cost < c_con_hinf.avg_cost
        fprintf('Percentage increase: SH2/SR: %5.2f   ', 100 * (c_con_h2.avg_cost   - reg_con_nc_unc.avg_cost) / reg_con_nc_unc.avg_cost)
        fprintf('SHinf/SR: %5.2f',                       100 * (c_con_hinf.avg_cost - reg_con_nc_unc.avg_cost) / reg_con_nc_unc.avg_cost)
    elseif c_con_h2.avg_cost < reg_con_nc_unc.avg_cost && c_con_h2.avg_cost < c_con_hinf.avg_cost
        fprintf('Percentage increase: SR/SH2: %5.2f   ', 100 * (reg_con_nc_unc.avg_cost - c_con_h2.avg_cost) / c_con_h2.avg_cost)
        fprintf('SHinf/SH2: %5.2f',                      100 * (c_con_hinf.avg_cost     - c_con_h2.avg_cost) / c_con_h2.avg_cost)
    else
        fprintf('Percentage increase: SR/SHinf: %5.2f   ', 100 * (reg_con_nc_unc.avg_cost - c_con_hinf.avg_cost) / c_con_hinf.avg_cost)
        fprintf('SH2/SHinf: %5.2f',                        100 * (c_con_h2.avg_cost       - c_con_hinf.avg_cost) / c_con_hinf.avg_cost)
    end
    fprintf('\n------------------------------------------------------\n\n')
    clear c_con_h2 c_con_hinf reg_con_nc_unc; % Clear variables corresponding to past disturbances profiles
    
end
clear i j w;
function [Irestored] = PDE_inpainting(I, mask)
% Optimizeid Image Inpainting using Heat Equation-based PDE
[imX, imY] = size(I);

dx = 1;  
dt = 0.2;   % Reduced for stability
t_max = 200; 

ts = 1:dt:t_max;
ts_n = numel(ts);

lambda = 0.4;
r = lambda * (dt / dx^2); 

% Use sparse matrices for efficient computation
L_X = spdiags([r * ones(imX,1), -2*r * ones(imX,1), r * ones(imX,1)], [-1, 0, 1], imX, imX);
L_Y = spdiags([r * ones(imY,1), -2*r * ones(imY,1), r * ones(imY,1)], [-1, 0, 1], imY, imY);

% Neumann Boundary Conditions
L_X(1,2) = 2*r;  L_X(imX,imX-1) = 2*r;
L_Y(1,2) = 2*r;  L_Y(imY,imY-1) = 2*r;

chi = double(mask);
U_old = double(I);

for k = 1:ts_n
    % Compute finite difference in both x and y directions efficiently
    U_new = U_old + L_X * U_old + U_old * L_Y + dt * (chi .* (I - U_old));
     U_old = U_new;
end

Irestored = U_new;

end

%% QG_solver_5b.m
% solves the QG barotropic equations
% follows Coiffier, Fundamentals of Numerical Weather Prediction
% this code is currently unvalidated and for demo purposes only

close all
clear all

%% set up spatial grid
%nx    = 256; % number of gridpoints in x direction
%ny    = 256; % number of gridpoints in y direction
nx    = 64; % number of gridpoints in x direction
ny    = 64; % number of gridpoints in y direction
Lx    = 1.0; % size of domain in x direction
Ly    = 1.0; % size of domain in y direction
dx    = Lx/nx;
dy    = Ly/ny;
x1    = (1:nx)*dx - Lx/2;
y1    = (1:ny)*dy - Ly/2;
[x,y] = meshgrid(x1,y1); x=x'; y=y';

%% set up time parameters and initial conditions
%nt    = 600; % number of timesteps
nt    = 600; % number of timesteps
dt    = 0.05; % timestep
nu    = 0.01; % value for spatial filter term
ze0   = 5*(rand(nx,ny) - 0.5); % random
delta = zeros(nx,ny); % flow divergence (zero for now)
tau   = 1e18; % damping timescale
beta  = 0.0; % planetary vorticity gradient [1/m/s]
fv    = 0.0 + beta*y; % planetary vorticity [1/s]

%% set up spatial operators
% double spatial derivative (del^2)
DD    = create_DD(dx,dy,nx,ny);
% centered spatial averages
Ax    = @(f) 0.5*[ (f(nx,:)+f(2,:))' (f(1:nx-2,:)+f(3:nx,:))' (f(1,:)+f(nx-1,:))' ]';
Ay    = @(f) 0.5*[2*f(:,1) (f(:,1:ny-2)+f(:,3:ny)) 2*f(:,ny)];
% centered spatial derivatives
Dx    = @(f) [ (f(2,:)-f(nx,:))' (f(3:nx,:)-f(1:nx-2,:))' (f(1,:)-f(nx-1,:))']'/(2*dx);
Dy    = @(f) [ (f(:,2)-f(:,ny))  (f(:,3:ny)-f(:,1:ny-2))  (f(:,1)-f(:,ny-1)) ] /(2*dy);

%% define main anon functions for temporal update
a1       = @(X2)     reshape(X2,[nx*ny 1]); % functions to flip between 1D and 2D arrays
a2       = @(X1)     reshape(X1,[nx ny]);
J1       = @(psi,ze) +Dx(psi).*Dy(ze) - Dy(psi).*Dx(ze); % the three jacobians
J2       = @(psi,ze) +Dx(psi.*Dy(ze)) - Dy(psi.*Dx(ze)); % formulation conserves ang. mom. (vorticity)
J3       = @(psi,ze) -Dx(ze.*Dy(psi)) + Dy(ze.*Dx(psi));
dzedt    = @(x,ze)   -(J1(x,ze)+J2(x,ze)+J3(x,ze))/3; % time derivative of ze
dzedt_sd = @(x,ze)   dzedt(x,ze) - ze.*delta - (ze-fv)/tau; % source and damping terms included
zeta     = @(psi)    a2(DD*a1(psi)) + fv; % get ze from psi
psi_i    = @(ze)     a2(DD\a1(ze)); % get psi from ze
dpsidt   = @(psi)    psi_i(dzedt_sd(zeta(psi),psi)); % time derivative of psi
spa_fil  = @(f)      (1-2*nu)*f  + nu*Ax(f) + nu*Ay(f); % perform horizontal spatial filtering
KE       = @(psi)    0.5*((Dy(psi)).^2 + (Dx(psi)).^2); % kinetic energy (prob. a simpler way?)
glob_av  = @(M)      sum(sum(M))*dx*dy/(Lx*Ly); % global average of quantity
dXdt     = @(X)      a1(dpsidt(a2(X))); % same as dpsidt but with X, vector

% specific to std iteration scheme
new_step = @(X,X_m) a1(spa_fil(a2(X_m + 2*dt*dXdt(X))));

%% do time iteration
X_m = a1(psi_i(ze0)); % get initial last-step streamfunction from vorticity
X   = X_m + dt*dXdt(X_m); % get initial streamfunction (as 1D state vector)
for it=1:nt
    
    X_t = X; % psi_0 is current value
    X   = new_step(X,X_m);
    X_m = X_t; % new psi_m is old value
    
    % display results
    if mod(it,100)==0
        it
        disp_results(x,y,zeta(a2(X))-fv,t_ar,K_ar,En_ar)
        pause(0.05)
    end
    
    % calculate global quantities
    t_ar(it)  = it*dt; % time [s]
    K_ar(it)  = glob_av(KE(a2(X))); % specific K.E. [m2/s2]
    En_ar(it) = glob_av(0.5*zeta(a2(X)).^2); % enstrophy [1/s2]
    
    if mod(it,5)==0
        ze_ar(:,:,floor(it/5)+1) = zeta(a2(X));
    end
    
end

if 0

vidObj = VideoWriter('QG_test.avi');
open(vidObj);
for k=1:size(ze_ar,3)
    f = figure('visible','off');
    surf(x,y,squeeze(ze_ar(:,:,k)));
    caxis([-1 1]*0.5)
    shading flat; axis equal; 
    axis off
    view(2)
    currFrame = getframe(f);
    writeVideo(vidObj,currFrame);
end
close(vidObj);

end

%% define standard functions
function DD = create_DD(dx,dy,nx,ny)

% double spatial derivative
a   = 1/dx^2;
b   = -2*(1/dx^2 + 1/dy^2);
c   = 1/dx^2;
DD1 = b*diag(ones(nx,1)) + c*diag(ones(nx-1,1),1) + a*diag(ones(nx-1,1),-1);
DD1(nx,1) = 1/dx^2;
DD1(1,nx) = 1/dx^2;
DD1       = sparse(DD1);
% periodic BCs in x-direction

DD = sparse(nx*ny,nx*ny);
for ia=1:ny
    for ib=1:ny
        if(ia==ib)
            % core diagonal
            a = (ia-1)*nx + 1;
            b = ia*nx;
            DD(a:b,a:b)=DD1;
        elseif(ia==ib+1 || ia==ib-1)
            % side diagonals (excluding periodic boundaries)
            a = (ia-1)*nx + 1;
            b = ia*nx;
            c = (ib-1)*nx + 1;
            d = ib*nx;
            DD(a:b,c:d) = +speye(nx)/dy^2;
        end
    end
end

return

end
function [] = disp_results(x,y,zz,t_ar,K_ar,En_ar)

figure(1)

subplot(2,1,1)
surf(x,y,zz);
shading flat; axis equal; colorbar vert
view(2)
xlabel('x')
ylabel('y')
title(['\zeta - f [1/s], t = ' num2str(t_ar(end),2) ' s'])

subplot(2,2,3)
semilogy(t_ar,K_ar,'r');
xlabel('t [s]')
ylabel('specific K.E. [m2/s2]')

subplot(2,2,4)
semilogy(t_ar,En_ar,'b');
xlabel('t [s]')
ylabel('En [1/s2]')

return

end



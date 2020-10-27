#!/usr/bin/env python
# Copyright (C) 2020 Christian Quirouette <cquir@ryerson.ca>
#                    Catherine Beauchemin <cbeau@users.sourceforge.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy 

# PDE parameters

beta = 1e-6 # rate at which cells become infected (units = TCID_50/(ml*h))
tau_E =	8.0 # length of eclipse phase (units = h)
tau_I = 10.0 # lifespan of productively infected cells (units = h)
p_0 = 1.0e7 # rate of virus production (units = TCID_50/(ml*h))
c_0 = 0.2 # rate at which virus is cleared (units = /h)
v_a = 40E-6*60.0*60.0 # advection speed (units = m/h)
D_PCF = 1e-12*60.0*60.0 # diffusion rate of virus in PCF (units = m^2/h)

# spatial and temporal parameters 

L = 0.3 # length of the RT (units = m)
Nx = 3000 # number of computational boxes
delta_x = L/Nx # distance step (units = m)
delta_t = delta_x/v_a # time step (units = h) 
Nt = int(round((10.0*24.0)/delta_t)) # number of time steps

# cell regeneration parameters

r_D = 0.75/24.0 # regeneration rate (units = /h)
tau_D = 1.0*24.0 # proliferation delay ( units = h)

# immune response parameters

lambda_gI = 2.0/24.0 # growth rate of IFN (units = /h)
lambda_dI = 1.0/24.0 # decay rate of IFN (units = /h)
IC_50 = 0.5 # amount of IFN required to reduce the viral production by half
t_pI = 3*24.0 # time of IFN peak (units = h)
alpha_i = 0.75/24.0 # growth rate of Ab (units = /h)
A_0 = 2.0e-3 # initial amount of Ab
k_v = 500.0 # Ab binding affinity 
t_pC = 8*24.0 # time of CTL peak
lambda_gC = 2.0/24.0 # growth rate of IFN (units = /h)
lambda_dC = 0.4/24.0 # decay rate of IFN (units = /h)
k_E = 50.0 # killing rate of infectious cells per CTL (units = /d)

# creating matrix to store [T E1 E2 ... EnE I1 I2 ... InI V] for all computational boxes at a given time step  

nE = 60 # number of eclipse phase compartments
nI = 60 # number of infectious phase compartments
Px = numpy.zeros((2+nE+nI,Nx)) 
T = 0
E1 = 1
I1 = 1+nE
V = (2+nE+nI)-1 

# creating matrix for dead cell populations 

m = int(round((tau_D)/delta_t)) # number of time steps corresponding to tau_D
if m == 0:
	D = numpy.zeros(Nx) # array or matrix to store dead cell population
else:
	D = numpy.zeros(((m+1),Nx)) 

# setting the virus population and target cell population at time t = 0

x_d = 0.15 # deposition depth (units = m)
sigma = 0.0005 # standard deviation of the virus distribution at time t = 0 (units = m)
divi = 2.0*sigma**2
V_0 = 1.0/numpy.mean(1.0/(numpy.sqrt(numpy.pi*divi))*numpy.exp(-((delta_x*numpy.arange(Nx)-x_d)**2)/divi)) # amount of viruses deposited (units = TCID_50/ml)
Px[V,:] = V_0/(numpy.sqrt(numpy.pi*divi))*numpy.exp(-((delta_x*numpy.arange(Nx)-x_d)**2)/divi)
Px[T,:] = 1.0 

# defining the coefficients matrices used for the Crank Nicolson scheme

alpha = (D_PCF*delta_t)/(delta_x**2)*numpy.ones(Nx)
# Build the Crank-Nicholson future (left-hand side) matrix
M = numpy.diag(-alpha[:-1],-1)+numpy.diag(2+2*alpha)+numpy.diag(-alpha[:-1],1)
# Enforce reflective boundary conditions at bottom (-alpha in diagonal term)
M[-1,-1] -= alpha[0]
M = numpy.matmul(numpy.linalg.inv(M),-M+numpy.diag(4*numpy.ones(Nx)))

# stuff that can be defined outside of for loop 

dtnEkE = numpy.zeros((nE+1,Nx))
dtnIdI = numpy.zeros((nI+1,Nx))	
tnext = -1; idx = -1
tprint = 0.5 # store only every tprint time the results (units = h)

# creating matrix to store [T E I V] for every time step 

Nprint = int(round((10.0*24.0)/tprint)) # number of time steps to print
Pt =  numpy.zeros((4,Nprint))
Pt[:,0] = [1,0,0,numpy.mean(Px[V,:])] # setting the inital values for T E I V


for n in range (1,Nt) :	
	
	# creating temp. variables using the previous time step values

	F = 2.0/(numpy.exp(-lambda_gI*(n*delta_t-t_pI))+numpy.exp(lambda_dI*(n*delta_t-t_pI)))
	p = p_0/(1.0+F/IC_50)
	A = 1.0/(1.0+(1.0/A_0-1.0)*numpy.exp(-alpha_i*n*delta_t))
	c = c_0+k_v*A
	C = 2.0/(numpy.exp(-lambda_gC*(n*delta_t-t_pC))+numpy.exp(lambda_dC*(n*delta_t-t_pC)))
	emcdt = numpy.exp(-c*delta_t)
	Vcst = p/c*(1.0-emcdt)
	dtbTV = delta_t*beta*Px[T,:]*Px[V,:]
	dtrdD = numpy.minimum(D[m,:],delta_t*r_D*Px[T,:]*D[numpy.mod(n,m),:])
	dtnEkE[1:int(nE/2),:] = delta_t*(nE*Px[1:int(nE/2),]/tau_E)
	dtnEkE[int(nE/2):nE+1,:] = delta_t*(nE*Px[int(nE/2):nE+1,]/tau_E+k_E*C*Px[int(nE/2):nE+1,:])
	dtnIdI[1:nI+1,:] = delta_t*(nI*Px[nE+1:nE+nI+1,:]/tau_I+k_E*C*Px[nE+1:nE+nI+1,:])
	Px[V,:] = numpy.roll(numpy.matmul(M,Px[V,:]),-1)
	Px[V,-1] = 0		
	dtpIcV = delta_t*(p*numpy.sum(Px[I1:I1+nI,:],axis=0)-c*Px[V,:])

	# solving the equations
		
	Px[T,:] -= (dtbTV-dtrdD)
	Px[E1,:] += dtbTV-dtnEkE[1]
	Px[2:nE+1,:]-= numpy.diff(dtnEkE[1:nE+2,:],axis=0)
	Px[I1,:] += dtnEkE[nE,:]-dtnIdI[1,:]
	Px[2+nE:nE+nI+1,:]-= numpy.diff(dtnIdI[1:nI+2,:],axis=0)
	Px[V,:] += dtpIcV
	D[numpy.mod(n,m),:] = D[m,:] 
	D[m,:] = 1-Px[T,:]-numpy.sum(Px[E1:(E1+nE),:],axis=0)-numpy.sum(Px[I1:(I1+nI),:],axis=0) 

	#storing the results
	if n*delta_t > tnext:
		tnext = n*delta_t+tprint	
		Tt = numpy.mean(Px[T,:]) 
		Et = numpy.mean(numpy.sum(Px[E1:(E1+nE),:],axis=0))
		It = numpy.mean(numpy.sum(Px[I1:(I1+nI),:],axis=0))
		Vt = numpy.mean(Px[V,:])
		idx += 1
		Pt[:,idx]=[Tt,Et,It,Vt]

#writing data to file
t = tprint*numpy.arange(Nprint)
data = numpy.vstack((t,Pt)).T
datafile_path = 'model-results.dat'
with open(datafile_path,'w') as datafile_id :
	numpy.savetxt(datafile_id, data, delimiter = ' ')

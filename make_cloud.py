import numpy
import math
import matplotlib.pyplot as plt
#my script extracted from molecular_cloud.py of amuse 
# https://amusecode.github.io/getting-started
# credit: Simon Portegies Zwart


#/////////////////////////UNITS/////////////////////////////

G=6.674e-8
msun=1.989e33 #grams
pc=3.086e+18 #cm
sec=31556952.0 #seconds in 1 year

#SCALES FOR PHYSICAL UNITS 
#The code calculates quantities in N-body units using for mass scale and length scale the total cloud mass and the cloud radius, respectively
unit_mass=4.3e4*msun #CLOUD MASS IN UNITS OF g
unit_length=10.*pc #CLOUD RADIUS IN cm 3.086e19/sqrt(3)
unit_time=(unit_length**3./G/unit_mass)**0.5 # in sec

#SCALES FOR FINAL N-BODY UNITS (I have chosen msun and pc)
unit_mass_new=msun #nbody mass = msun
unit_length_new=pc # nbody length = pc
unit_time_new=(unit_length_new**3./G/unit_mass_new)**0.5 # in sec   
print("Time unit in Myr= ", unit_time_new/3.1536e7/1e6) 


#/////////////////////////FUNCTIONS/////////////////////////////
#from inverse fast fourier transform to real numbers
#we are starting from a power spectrum -> need to find it in the real space 
def make_ifft_real(nf,vi): 
    if vi.ndim==3:
# body of cube
        vi[1:nf,1:2*nf,1:2*nf]=numpy.conj(vi[2*nf-1:nf:-1,2*nf-1:0:-1,2*nf-1:0:-1])

# 3 lower + middle planes
        vi[0,1:nf,1:2*nf]=numpy.conj(vi[0,2*nf-1:nf:-1,2*nf-1:0:-1]) #Return the complex conjugate
        vi[1:nf,0,1:2*nf]=numpy.conj(vi[2*nf-1:nf:-1,0,2*nf-1:0:-1])
        vi[1:nf,1:2*nf,0]=numpy.conj(vi[2*nf-1:nf:-1,2*nf-1:0:-1,0])
        vi[nf,1:nf,1:2*nf]=numpy.conj(vi[nf,2*nf-1:nf:-1,2*nf-1:0:-1])

# 7 lines
        vi[0,0,1:nf]=numpy.conj(vi[0,0,2*nf-1:nf:-1])
        vi[0,1:nf,0]=numpy.conj(vi[0,2*nf-1:nf:-1,0])
        vi[1:nf,0,0]=numpy.conj(vi[2*nf-1:nf:-1,0,0])
        vi[0,nf,1:nf]=numpy.conj(vi[0,nf,2*nf-1:nf:-1])
        vi[nf,0,1:nf]=numpy.conj(vi[nf,0,2*nf-1:nf:-1])
        vi[nf,nf,1:nf]=numpy.conj(vi[nf,nf,2*nf-1:nf:-1])
        vi[nf,1:nf,0]=numpy.conj(vi[nf,2*nf-1:nf:-1,0])

# 8 points
        vi[0,0,0]=2*numpy.real(vi[0,0,0])
        vi[nf,0,0]=2*numpy.real(vi[nf,0,0])
        vi[0,nf,0]=2*numpy.real(vi[0,nf,0])
        vi[nf,nf,0]=2*numpy.real(vi[nf,nf,0])

        vi[0,0,nf]=2*numpy.real(vi[0,0,nf])
        vi[nf,0,nf]=2*numpy.real(vi[nf,0,nf])
        vi[0,nf,nf]=2*numpy.real(vi[0,nf,nf])
        vi[nf,nf,nf]=2*numpy.real(vi[nf,nf,nf])
        return vi
    
    return -1  
  
#generate the random gaussian velocity field on the grid
def random_field(nf=32, power=-4., seed=None):
    if seed is not None:
        numpy.random.seed(seed)
    
    freq=numpy.mgrid[-nf:nf,-nf:nf,-nf:nf]   

    fi,fj,fk=freq

    fi=fi.flatten()
    fj=fj.flatten()
    fk=fk.flatten()
    
    norm=-numpy.log(numpy.random.uniform(0.,1.,len(fi)))*(fi**2+fj**2+fk**2+1.e-30)**(power/4.)
    phase=numpy.random.uniform(0.,1.,len(fi))*2*numpy.pi
    vi=norm*numpy.exp(phase*1j)

    vi=vi.reshape(nf*2,nf*2,nf*2)

    vi[nf,nf,nf]=0.
    
    vi=make_ifft_real(nf,vi)
    #transforms previous complex numbers to real
    
    vi=numpy.fft.ifftshift( vi)
    #fft.fftshift Shift the zero-frequency component to the center of the spectrum. fft.ifftshift does the inverse

    vi=numpy.fft.ifftn(vi)
    #Compute the N-dimensional inverse discrete Fourier Transform.
    
    if vi.imag.max()>1.e-16:
        print("check random field")
    return vi

#make sure that the velocity field is divergence free
def make_div_free(nf,vx,vy,vz):    
    vx=numpy.fft.fftn(vx)
    vx=vx.flatten()
    vy=numpy.fft.fftn(vy)
    vy=vy.flatten()
    vz=numpy.fft.fftn(vz)
    vz=vz.flatten()

    freq=numpy.mgrid[-nf:1.*nf,-nf:1.*nf,-nf:1.*nf] 
    fi,fj,fk=freq
    fi=numpy.fft.fftshift( fi)
    fj=numpy.fft.fftshift( fj)
    fk=numpy.fft.fftshift( fk)

    fi=fi.flatten()
    fj=fj.flatten()
    fk=fk.flatten()
    ff=fi*fi+fj*fj+fk*fk+1.e-30
    
    vdotf=(vx*fi+vy*fj+vz*fk)
    vx=vx-fi*vdotf/ff
    vy=vy-fj*vdotf/ff
    vz=vz-fk*vdotf/ff

    del fi,fj,fk,ff

    vx=vx.reshape(2*nf,2*nf,2*nf)
    vy=vy.reshape(2*nf,2*nf,2*nf)
    vz=vz.reshape(2*nf,2*nf,2*nf)

# zero out nyquist freq planes: strictly speaking this is too drastic....
# inside the nyquist planes only v// f x f_mirror needs to be enforced (methinks) 
    vx[nf,0:2*nf,0:2*nf]=0.
    vx[0:2*nf,nf,0:2*nf]=0.
    vx[0:2*nf,0:2*nf,nf]=0.
    vy[nf,0:2*nf,0:2*nf]=0.
    vy[0:2*nf,nf,0:2*nf]=0.
    vy[0:2*nf,0:2*nf,nf]=0.
    vz[nf,0:2*nf,0:2*nf]=0.
    vz[0:2*nf,nf,0:2*nf]=0.
    vz[0:2*nf,0:2*nf,nf]=0.

    vx=numpy.fft.ifftn(vx)
    vy=numpy.fft.ifftn(vy)
    vz=numpy.fft.ifftn(vz)

    if vx.imag.max()>1.e-16:
        print("check div-free field")
    if vy.imag.max()>1.e-16:
        print("check div-free field")
    if vz.imag.max()>1.e-16:
        print("check div-free field")

    return vx.real,vy.real,vz.real

#/////////////////////////CLASSES/////////////////////////////
#generate uniformly spaced points in cube
class uniform_random_unit_cube(object):
    def __init__(self,targetN):
        self.targetN=targetN
        self.par=int(targetN) #long(targetN)
    def make_xyz(self):
        x=numpy.random.uniform(-1.,1.,self.par)
        y=numpy.random.uniform(-1.,1.,self.par)
        z=numpy.random.uniform(-1.,1.,self.par)
        return x,y,z

#generate uniformly spaced points in unit sphere    
class uniform_unit_sphere(object):
    def __init__(self,targetN, base_grid=None):
        cube_sphere_ratio=4/3.*numpy.pi*0.5**3
        self.targetN=targetN
        self.estimatedN=targetN/cube_sphere_ratio
        if base_grid is None:
            self.base_grid=uniform_random_unit_cube
        else:
            self.base_grid=base_grid
   
    def cutout_sphere(self,x,y,z):
        r=x**2+y**2+z**2
        selection=r < numpy.ones_like(r)        
        x=x.compress(selection)
        y=y.compress(selection)
        z=z.compress(selection)
        return x,y,z

    def make_xyz(self):
        if(self.base_grid==uniform_random_unit_cube):
            estimatedN=self.estimatedN
            x=[]
            while len(x) < self.targetN:
                estimadedN=estimatedN*1.1+1
                x,y,z=self.cutout_sphere(*(self.base_grid(estimatedN)).make_xyz())
            return x[0:self.targetN],y[0:self.targetN],z[0:self.targetN]  
        else:
            return self.cutout_sphere(*(self.base_grid(self.estimatedN)).make_xyz())

#interpolate values from the grid to the particles
def interpolate_trilinear(x,y,z,farray):

    if farray.ndim!=3:
        return -1
      
    nx,ny,nz=farray.shape    
    dx=2./nx
    dy=2./ny
    dz=2./nz

    fx,xint=numpy.modf((x+1)/dx)
    fy,yint=numpy.modf((y+1)/dy)
    fz,zint=numpy.modf((z+1)/dz)

    xint=xint.astype('i')
    yint=yint.astype('i')
    zint=zint.astype('i')

    xint1=numpy.mod(xint+1,nx)
    yint1=numpy.mod(yint+1,nx)
    zint1=numpy.mod(zint+1,nx)

    q111 = farray[xint, yint, zint]
    q211 = farray[xint1, yint, zint]
    q221 = farray[xint1, yint1, zint]
    q121 = farray[xint, yint1, zint]
    q112 = farray[xint, yint, zint1]
    q212 = farray[xint1, yint, zint1]
    q222 = farray[xint1, yint1, zint1]
    q122 = farray[xint, yint1, zint1]

    return (q222* fx*fy*fz +  
      q122* (1-fx)*fy*fz +  
      q212* fx*(1-fy)*fz +  
      q112* (1-fx)*(1-fy)*fz +  
      q221* fx*fy*(1-fz) +  
      q121* (1-fx)*fy*(1-fz) +  
      q211* fx*(1-fy)*(1-fz) +  
      q111* (1-fx)*(1-fy)*(1-fz))


#construct the object molecular, calling all the other functions and classes defined before 
class molecular_cloud(object):

#	set the main parameters for the cloud
    def __init__(self,nf=32,power=-4.,targetN=10000,  ethep_ratio=0.01,ekep_ratio=1.,seed=None,base_grid=None):
        #self is used to represent the instance of the class
        self.nf=nf
        self.power=power
        self.targetN=targetN
        self.seed=seed
        self.base_grid=base_grid
        self.ethep_ratio=ethep_ratio
        self.ekep_ratio=ekep_ratio
        print("nf, power, targetN, seed\n")     
        print(nf, power, targetN, seed)

#	generate the cloud accordingly
    def new_model(self):
        if self.seed is not None:
            numpy.random.seed(self.seed)
            
        #creates random field of velocities following turbulence power spectrum
        vx_field=random_field(self.nf,self.power) 
        vy_field=random_field(self.nf,self.power)
        vz_field=random_field(self.nf,self.power)
        print('\nself.nf, self.power, self.seed\n')        
        print(self.nf, self.power, self.seed)
        #print vx_field

        #makes the velocity field divergence free
        vx_field,vy_field,vz_field=make_div_free(self.nf,vx_field,vy_field,vz_field)

        #creates particle positions uniform in the sphere
        base_sphere=uniform_unit_sphere(self.targetN,base_grid=self.base_grid)
        x,y,z=base_sphere.make_xyz()
        self.actualN=len(x)
        
        #interpolates from the velocity field grid
        #the velocity at the position of each single particle
        #similar to what we have done for NFW halos
        vx=interpolate_trilinear(x,y,z,vx_field)
        vy=interpolate_trilinear(x,y,z,vy_field)
        vz=interpolate_trilinear(x,y,z,vz_field)
        mass=numpy.ones_like(x)/self.actualN


        #defines potential and kinetic energy
        #starting from ethep_ratio and ekep_ratio
        Ep=3./5.
        self.internalE=Ep*self.ethep_ratio
        Ek=0.5*mass[0]*(vx**2+vy**2+vz**2).sum()
        vfac=math.sqrt(self.ekep_ratio*Ep/Ek)
        vx=vx*vfac
        vy=vy*vfac
        vz=vz*vfac
        Ek=0.5*mass[0]*(vx**2+vy**2+vz**2).sum()

        internal_energy=numpy.ones_like(x)*self.internalE

        #convert to physical units
        mass=mass*unit_mass
        x=x*unit_length
        y=y*unit_length
        z=z*unit_length
        vx=vx*unit_length/unit_time
        vy=vy*unit_length/unit_time
        vz=vz*unit_length/unit_time

      
        #convert to nbody units (msun and pc)
        #for gasoline
        #you can use different N-body units
        mass=mass/unit_mass_new
        x=x/unit_length_new
        y=y/unit_length_new
        z=z/unit_length_new
        vx=vx*unit_time_new/unit_length_new
        vy=vy*unit_time_new/unit_length_new
        vz=vz*unit_time_new/unit_length_new

#  make sure that the CM is in the origin of the reference frame. Important for calculations, in this case!!     
        xcm=0.0	
        ycm=0.0
        zcm=0.0
        vxcm=0.0
        vycm=0.0
        vzcm=0.0

        xcm=sum(mass*x)
        ycm=sum(mass*y)
        zcm=sum(mass*z)
        vxcm=sum(mass*vx)
        vycm=sum(mass*vy)
        vzcm=sum(mass*vz)
        xcm=xcm/mass.sum()
        ycm=ycm/mass.sum()
        zcm=zcm/mass.sum()
        vxcm=vxcm/mass.sum()
        vycm=vycm/mass.sum()
        vzcm=vzcm/mass.sum()
        #print(xcm,ycm,zcm,vxcm,vycm,vzcm)
        x-=xcm
        y-=ycm
        z-=zcm
        vx-=vxcm
        vy-=vycm
        vz-=vzcm


        return (mass,x,y,z,vx,vy,vz)


##########MAIN###############

N = 10000

x = molecular_cloud(nf=64,power=-4.,targetN=N, ethep_ratio=0.01,ekep_ratio=1.,seed=39113,base_grid=None)
#nf (number of grid cells to sample the velocities)
#power (power spectrum slope for turbulence -4 is Burgers, -11/3. is Kolmogorov)
#targetN (number of particles in the output file = initial condition file for your simulation)

mass,x,y,z,vx,vy,vz= x.new_model()
#new_model() calculates the properties of the molecular cloud object

#print ascii output (for check)
ff=open('init_cond_MC.ascii','w')
ff.write("M [Msun] x [pc] y [pc] z [pc] vx [pc/timescale]  vy [pc/timescale]  vz [pc/timescale]\n")
for i in range(N):
    ff.write(str(mass[i])+' '+str(x[i])+' '+str(y[i])+' '+str(z[i])+' '+str(vx[i])+' '+str(vy[i])+' '+str(vz[i])+'\n')

#print tipsy ascii output
f=open('init_cond_MC.tip','w')
f.write(str(N)+' '+str(N)+' '+str(0)+'\n')
f.write(str(3)+'\n')
f.write(str(0.0)+'\n')

for i in range(N):
    f.write(str(mass[i])+'\n')
for i in range(N):
    f.write(str(x[i])+'\n')
for i in range(N):
    f.write(str(y[i])+'\n')
for i in range(N):
    f.write(str(z[i])+'\n')
for i in range(N):
    f.write(str(vx[i])+'\n')
for i in range(N):
    f.write(str(vy[i])+'\n')
for i in range(N):
    f.write(str(vz[i])+'\n')
for i in range(N):
    f.write(str(0.0)+'\n')  #density (if zero the code calculates it)
for i in range(N):
    f.write(str(10.0)+'\n') #temperature/K
for i in range(N):
    f.write(str(1e-2*pc/unit_length_new)+'\n') #softening length
    #(10^-2 pc)
for i in range(N):
    f.write(str(0.02)+'\n') #metal gas Z = 0.02 is approx solar
for i in range(N):
    f.write(str(0.0)+'\n') #potential energy (if zero the code calculates it)

plt.scatter(x,y,s=1)
plt.xlabel("x [pc]")
plt.ylabel("y [pc]")
plt.show()

vx=vx*unit_length_new/unit_time_new/1e5
vy=vy*unit_length_new/unit_time_new/1e5
plt.scatter(vx,vy,s=1)
plt.xlabel("v$_x$ [km/s]")
plt.ylabel("v$_y$ [km/s]")
plt.show()

plt.scatter(x,vx,s=1)
plt.xlabel("x [pc]")
plt.ylabel("v$_x$ [km/s]")
plt.show()


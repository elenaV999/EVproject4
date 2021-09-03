#HISTOGRAM OF DENSITY DISTRIBUTION IN TIME
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import astropy.constants as const
import astropy.units as u

def readfast(fname, N):		
	f=open(fname,"r")		#open file "fname" for reading
	a = np.zeros(N,dtype="float")	#arrays of N elements=0.0 of type float
	i=0
	for linetext in f:			
		word_list = linetext.split()	#splits a line in elements
		a[i]=np.float(word_list[0])		 #fill numpy array with float data from the selected columns
		i=i+1
	f.close()
	return (a)
   
   
#constants
Lscale=const.pc.value	#parsec in m
Mscale=const.M_sun.value  #Msun  in kg
tscale=14.92
print('Lscale= ',Lscale, '\nMscale= ', Mscale)


mu=2.46 
G=const.G.value*(u.m**3/u.kg/u.s**2)
kb = const.k_B.value*(u.m**2*u.kg/u.s**2/u.K) 
mp = const.m_p.value*u.kg


timescale=np.sqrt(Lscale**3/(G.value*Mscale))*u.s
timescale=timescale.to(u.Myr)
timescale=timescale.value
print('t_scale = ',timescale, 'Myr' )

c_mj=kb/mu/mp/G  
c_mj = c_mj.to(u.M_sun/u.K/u.pc)
c_mj=c_mj.value

c_n=mu*mp.to(u.g)
c_n=c_n.value



#DENSITY ON A GRID
Nbins=40 #n° of bins in the grid

nmin=1e-2
nmax=5e13
tt_=np.arange(0.1,4.1,0.1)
bins=np.logspace(np.log10(nmin),np.log10(nmax), Nbins)
print(bins)

tt, yy = np.meshgrid(tt_ , bins) 	#shape=(N,N)   
zz = np.zeros((Nbins, Nbins), int) 

if np.shape(yy)!=np.shape(zz) :  #check if z and (x,y) grid shapes are compadible
	print("ERROR: grid and data matrix of different shape. Exit program")
	exit()



# READ GASOLINE FILES  
Nfiles=40   
iOutInterval = 1 
num = np.arange(1, Nfiles+1, 1 ) #start, stop, step
nfile = np.char.zfill(num.astype('str'),6)   #number in the file name


Nlines = 120003
for k in range(Nfiles):	#loop over all files
	fname="out."+str(nfile[k])+".ascii"  
	print(fname)  
	a=readfast(fname, Nlines)
	t=a[2]
	aa = a[3:]
	m,x,y,z,vx,vy,vz,dens,T,h,Z,pot = np.array_split(aa,12)
	
	t=t*tscale #time in Myr
	n=dens*Mscale*1e3/(Lscale*1e2)**3 
	n=n/c_n   #n°dens in 1/cm^3
	

	h, b = np.histogram(n, bins=bins)#Nbins, range=(np.log10(nmin), np.log10(nmax) ))
	for l in range(Nbins-1) :
		zz[l][k] = h[l]*4.3  	#k=1,2,.....40  time 	/ l=n° of particles with that density
							#mass on the z axis (each particle is 4.3 M_sun)

### contour plot
lev=np.logspace(0, 5, num=16, base=10) 
cs = plt.contourf( tt, yy, zz, cmap=cm.viridis, norm=colors.SymLogNorm(linthresh=0.5), levels=lev, extend='min') 

ticks=np.logspace(0, 5, num=6, base=10) 
cbar= plt.colorbar(cs , orientation='vertical', ticks=ticks, extendrect=True, extendfrac=0.02) #
cbar.set_label("Mass [ $M_{\odot}$ ]", fontsize = 13)

plt.xlabel("t [Myr]",  fontsize= 13 ) 
plt.ylabel("n [cm$^{-3}$]",  fontsize= 13 ) 
plt.yscale('log')
plt.show()














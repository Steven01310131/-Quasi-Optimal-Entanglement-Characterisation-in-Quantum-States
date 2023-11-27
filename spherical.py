import numpy as np
import matplotlib.pyplot as plt
import astropy
import cmath
def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
def scatterplot_sphere_points(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot of points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=12)
    # Surface plot of the sphere
    phi, theta = np.mgrid[-np.pi:np.pi:100j, -np.pi/2:np.pi/2:50j]
    x_sphere = np.cos(theta) * np.cos(phi)
    y_sphere = np.cos(theta) * np.sin(phi)
    z_sphere = np.sin(theta)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color = 'g',alpha = 0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])])
    ax.view_init(elev=20, azim=45)
    ax.dist = 10
    plt.show()
def Spoints(N):
    k = np.arange(1, N+1)

    h = -1 + 2/(N-1) * (k-1)
  
  
    phi = np.arccos(h) - np.pi/2
    
    theta = np.zeros(N)
    theta[0] = 0
    for j in range(1, N-1):
        theta[j] = (theta[j-1] + 3.6/np.sqrt(N) * 1/np.sqrt(1 - h[j]**2)) % (2*np.pi)- np.pi
    val = cmath.cos(complex(0,theta[100])) 
    print(val)
    theta[N-1] = 0
    # theta = theta - np.pi
    x,y,z = sph2cart(theta,phi,1)
    X = np.column_stack((x,y,z))
    X[0, :] = [0, 0, 1]
    X[-1, :] = [0, 0, -1]
    return X
def to_spherical():
    pass

# Main code
X = Spoints(500)
scatterplot_sphere_points(X)
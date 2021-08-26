import odl
import numpy as np
from customOperators import *

#def residual(R, b):
#    def residual_functional(x, b=b, R=R):
#        val = np.linalg.norm(b.data.ravel()-R(x).data.ravel())/np.linalg.norm(b.data.ravel())
#        return val
#    return residual_functional

def TVRecon(sinogram, geometry, reconstruction_space, lamda):
    # Create the forward operator
    ray_trafo = odl.tomo.RayTransform(reconstruction_space, geometry)
    
    # Initialize gradient operator
    gradient = odl.Gradient(reconstruction_space)

    # Column vector of two operators
    op = odl.BroadcastOperator(ray_trafo, gradient)

    # Do not use the f functional, set it to zero.
    f = odl.solvers.ZeroFunctional(op.domain)

    # Create functionals for the dual variable

    # l2-squared data matching
    l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(sinogram)

    # Isotropic TV-regularization i.e. the l1-norm
    l1_norm = lamda * odl.solvers.L1Norm(gradient.range)#

    # Combine functionals, order must correspond to the operator K
    g = odl.solvers.SeparableSum(l2_norm, l1_norm)

    # --- Select solver parameters and solve using PDHG --- #

    # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
    op_norm = 1.1 * odl.power_method_opnorm(op)

    niter = 200  # Number of iterations
    #not necessarily optimal choice but gaurantess convergence!
    sigma = 1/(1.1*op_norm)
    tau = 1/(1.1*op_norm)

    # Optionally pass callback to the solver to display intermediate results
    #callback = (odl.solvers.CallbackShowConvergence(residual(ray_trafo, sinogram)))

    # Choose a starting point
    x = op.domain.zero()

    # Run the algorithm
    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)#, callback=callback)
    return x
    

def TVReconTime(sinogram, reconstruction_space, lamda_1, lamda_2, frames, thetas_all, angles_pr_frame, angles_pr_pi, detector_partition):

    #divide into angles to use pr. frame
    thetas = np.lib.stride_tricks.sliding_window_view(thetas_all, angles_pr_frame)
    
    #Strip time dimension
    reco_space_frame = odl.uniform_discr(reconstruction_space.min_pt[:-1], reconstruction_space.max_pt[:-1], reconstruction_space.shape[:-1])
    
    As = []
    for t in range(frames):
        angle_partition = odl.uniform_partition(thetas[t][0], thetas[t][0]+np.pi/(angles_pr_pi/angles_pr_frame), angles_pr_frame)

        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        As.append(odl.tomo.RayTransform(reco_space_frame, geometry))
   

    TimeOp = RadonTime(*As, domain=reconstruction_space)
    
    split = np.lib.stride_tricks.sliding_window_view(sinogram,angles_pr_frame, axis=0)
    sinogram_spaces = TimeOp.range.element(np.transpose(split[0:frames],(0,2,1)))
    
    gradient = odl.Gradient(TimeOp.domain)
    


    #Then lamda weighting - last dimension of first axis is time
    weights = np.ones(gradient.range.shape)
    weights[-1] = weights[-1]*lamda_2
    weight_vec = gradient.range.element(weights)
    weighted_gradient = odl.OperatorLeftVectorMult(gradient, weight_vec)


    L = odl.BroadcastOperator(TimeOp,weighted_gradient)
    f = odl.solvers.ZeroFunctional(L.domain)
    norm_2 = odl.solvers.L2NormSquared(TimeOp.range).translated(sinogram_spaces)


    norm_1 = lamda_1*odl.solvers.L1Norm(weighted_gradient.range)
    g = odl.solvers.SeparableSum(norm_2,norm_1)

    op_norm = odl.power_method_opnorm(L)

    #not necessarily optimal choice but gaurantess convergence!
    sigma = 1/(1.1*op_norm)
    tau = 1/(1.1*op_norm)

    x = L.domain.zero()

    # Run the algorithm
    odl.solvers.pdhg(x, f, g, L, niter=200, tau=tau, sigma=sigma)

    return x, sinogram_spaces

def TVReconTimeCompression(sinogram, reconstruction_space, lamda_1, lamda_2, frames, thetas_all, angles_pr_frame, angles_pr_pi, detector_partition, p=1.7):
    
        #divide into angles to use pr. frame
    thetas = np.lib.stride_tricks.sliding_window_view(thetas_all, angles_pr_frame)
    
    #Strip time dimension
    reco_space_frame = odl.uniform_discr(reconstruction_space.min_pt[:-1], reconstruction_space.max_pt[:-1], reconstruction_space.shape[:-1])
    
    As = []
    for t in range(frames):
        angle_partition = odl.uniform_partition(thetas[t][0], thetas[t][0]+np.pi/(angles_pr_pi/angles_pr_frame), angles_pr_frame)

        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        As.append(odl.tomo.RayTransform(reco_space_frame, geometry))
   

    TimeOp = RadonTime(*As, domain=reconstruction_space)
    
    split = np.lib.stride_tricks.sliding_window_view(sinogram,angles_pr_frame, axis=0)
    sinogram_spaces = TimeOp.range.element(np.transpose(split[0:frames],(0,2,1)))
    
    gradient = odl.Gradient(TimeOp.domain)


    #W = odl.operator.default_ops.IdentityOperator(TimeOp.domain)

    #ifft = odl.trafos.DiscreteFourierTransform(TimeOp.domain)
    #ones = np.ones(ifft.domain.shape)
    #keep_fraction = 0.3
    #r,c,t = multiphantom.shape
    #ones[int(r*keep_fraction):int(r*(1-keep_fraction)),:,:]=0
    #ones[:,int(c*keep_fraction):int(c*(1-keep_fraction)),:]=0
    #ones[:,:,int(c*keep_fraction):int(c*(1-keep_fraction))]=0
    #keepers = ifft.domain.element(ones)
    #weighted_ifft = odl.OperatorRightVectorMult(ifft, keepers)
    #W = weighted_ifft

    wavelet = odl.trafos.WaveletTransform(TimeOp.domain, wavelet='haar', nlevels=4)
    scales = wavelet.scales()
    iwavelet = wavelet.inverse * (1 / (np.power(p, scales)))
    W = iwavelet

    # Create regularizer as l1 norm

    #Then lamda weighting
    weights = np.ones(gradient.range.shape)
    weights[2,:,:,:] = weights[2,:,:]*lamda_2
    weight_vec = gradient.range.element(weights)
    weighted_gradient = odl.OperatorLeftVectorMult(gradient, weight_vec)

    TimeOp_comp = odl.operator.operator.OperatorComp(TimeOp,W)
    weighted_gradient_comp = odl.operator.operator.OperatorComp(weighted_gradient,W)

    L = odl.BroadcastOperator(TimeOp_comp,weighted_gradient_comp)
    f = odl.solvers.ZeroFunctional(L.domain)
    norm_2 = odl.solvers.L2NormSquared(TimeOp_comp.range).translated(sinogram_spaces)


    norm_1 = lamda_1*odl.solvers.L1Norm(weighted_gradient_comp.range)
    g = odl.solvers.SeparableSum(norm_2,norm_1)

    op_norm = odl.power_method_opnorm(L)

    #not necessarily optimal choice but gaurantess convergence!
    sigma = 1/(1.1*op_norm)
    tau = 1/(1.1*op_norm)

    x = L.domain.zero()
    # Optionally pass callback to the solver to display intermediate results
    #callback = (odl.solvers.CallbackPrintIteration(step=10) &
    #            odl.solvers.CallbackShow(step=10))

    # Run the algorithm
    odl.solvers.pdhg(x, f, g, L, niter=200, tau=tau, sigma=sigma)
      #               callback=callback)

    return W(x), sinogram_spaces

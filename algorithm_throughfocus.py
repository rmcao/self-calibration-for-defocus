# Description:
#  Modified from Michael Chen's code base structure
# Created by Ruiming Cao on May 09, 2020
# Contact: rcao@berkeley.edu
# Website: https://rmcao.net


import os
import numpy as np
import tensorflow as tf
from math import factorial

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

naxis = np.newaxis
pi = np.pi
np_complex_datatype = np.complex128

def cartToNa(point_list_cart, z_offset=0):
    """Function which converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)


def cart2Pol(x, y):
    rho          = (x * np.conj(x) + y * np.conj(y))**0.5
    theta        = np.arctan2(np.real(y), np.real(x)).astype(np_complex_datatype)
    return rho, theta


def genZernikeAberration(shape, pixel_size, NA, wavelength, z_coeff = [1], z_index_list = [0], fx_illu=0.0, fy_illu=0.0):
    assert len(z_coeff) == len(z_index_list), "number of coefficients does not match with number of zernike indices!"

    pupil             = genPupil(shape, pixel_size, NA, wavelength, fx_illu=fx_illu, fy_illu=fy_illu)
    fxlin             = _genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True) - fx_illu
    fylin             = _genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True) - fy_illu
    fxlin             = np.tile(fxlin[np.newaxis,:], [shape[0], 1])
    fylin             = np.tile(fylin[:, np.newaxis], [1, shape[1]])
    rho, theta        = cart2Pol(fxlin, fylin)
    rho[:, :]        /= NA/wavelength

    def zernikePolynomial(z_index):
        n                    = int(np.ceil((-3.0 + np.sqrt(9+8*z_index))/2.0))
        m                    = 2*z_index - n*(n+2)
        normalization_coeff  = np.sqrt(2 * (n+1)) if abs(m) > 0 else np.sqrt(n+1)
        azimuthal_function   = np.sin(abs(m)*theta) if m < 0 else np.cos(abs(m)*theta)
        zernike_poly         = np.zeros([shape[0], shape[1]], dtype = np_complex_datatype)
        for k in range((n-abs(m))//2+1):
            zernike_poly[:, :]  += ((-1)**k * factorial(n-k))/ \
                                    (factorial(k)*factorial(0.5*(n+m)-k)*factorial(0.5*(n-m)-k))\
                                    * rho**(n-2*k)

        return normalization_coeff * zernike_poly * azimuthal_function

    for z_coeff_index, z_index in enumerate(z_index_list):
        zernike_poly = zernikePolynomial(z_index)

        if z_coeff_index == 0:
            zernike_aberration = np.array(z_coeff).ravel()[z_coeff_index] * zernike_poly
        else:
            zernike_aberration += np.array(z_coeff).ravel()[z_coeff_index] * zernike_poly

    return zernike_aberration * pupil

def genPupil(shape, pixel_size, NA, wavelength, fx_illu = 0.0, fy_illu = 0.0, NA_in = 0.0):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = np.fft.ifftshift(_genGrid(shape[1],1/pixel_size/shape[1]))
    fylin        = np.fft.ifftshift(_genGrid(shape[0],1/pixel_size/shape[0]))
    pupil_radius = NA/wavelength
    pupil        = np.asarray((fxlin[naxis,:] - fx_illu)**2 + (fylin[:,naxis] - fy_illu)**2 <= pupil_radius**2)
    if NA_in != 0.0:
        pupil[(fxlin[naxis,:] - fx_illu)**2 + (fylin[:,naxis]-fy_illu)**2 < pupil_radius**2] = 0.0
    return pupil

def propKernel(shape, pixel_size, wavelength, prop_distance, NA = None, RI = 1.0, fx_illu=0.0, fy_illu=0.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = np.fft.ifftshift(_genGrid(shape[1],1/pixel_size/shape[1]))
    fylin        = np.fft.ifftshift(_genGrid(shape[0],1/pixel_size/shape[0]))
    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop = genPupil(shape, pixel_size, NA, wavelength, fx_illu = fx_illu, fy_illu = fy_illu)
    else:
        Pcrop = 1.0
    prop_kernel = Pcrop * np.exp(1j*2.0*pi*np.abs(prop_distance)*Pcrop*((RI/wavelength)**2 - (fxlin[naxis,:] - fx_illu)**2 - (fylin[:,naxis] - fy_illu)**2)**0.5)
    prop_kernel = prop_kernel.conj() if prop_distance < 0 else prop_kernel
    return prop_kernel

def _genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        kx      - 1D Fourier grid

    """
    xlin = (np.arange(size,dtype='complex128') - size//2) * dx
    if flag_shift:
        xlin = np.roll(xlin, (size)//2)
    return xlin


class ThroughFocusSolver:
    def __init__(self, imgs, pixel_size, wavelength, NA, z_planes, fx_illu=0.0, fy_illu=0.0, pad=(True,32),
                 zernike_order_min=0, zernike_order_max=9):
        self.amplitude     = imgs**0.5
        self.shape         = imgs[0].shape
        self.xlin          = _genGrid(self.shape[1], pixel_size)
        self.ylin          = _genGrid(self.shape[0], pixel_size)
        if len(pad) == 2:
            self.pad           = (pad[0],(pad[1],pad[1]))
        elif len(pad) == 3:
            self.pad           = (pad[0],(pad[1],pad[2]))

        self.RI_measure    = 1.0
        self.NA            = NA
        self.wavelength    = wavelength
        self.pixel_size    = pixel_size
        self.fx_illu       = fx_illu
        self.fy_illu       = fy_illu

        dim0           = self.shape[0] + 2*self.pad[1][0]
        dim1           = self.shape[1] + 2*self.pad[1][1]

        self.fxlin         = np.fft.ifftshift(_genGrid(dim1, 1 / pixel_size / dim1))
        self.fylin         = np.fft.ifftshift(_genGrid(dim0, 1 / pixel_size / dim0))

        self.pupil_mask = genPupil((dim0, dim1), self.pixel_size, self.NA, self.wavelength, fx_illu=self.fx_illu,
                         fy_illu=self.fy_illu).astype(np.complex64)

        self.zernike_order_min        = zernike_order_min
        self.zernike_order_max        = zernike_order_max
        self.zernike_coeffs   = np.zeros(zernike_order_max - zernike_order_min + 1).astype(np.float32)
        self.zernike_indices  = np.arange(self.zernike_order_min, self.zernike_order_max + 1)
        self.zernike_bases    = np.array([genZernikeAberration([dim0, dim1], self.pixel_size, self.NA, self.wavelength,
                                                      z_coeff=[1], z_index_list=[i], fx_illu=self.fx_illu,
                                                               fy_illu=self.fy_illu) for i in self.zernike_indices]).astype(np.complex64)
        self.pupil = None
        self.set_pupil()

        self.z_planes_tf = []
        self.set_z_pos(z_planes)
        # self.prop_kern_tf = tf.stack(tf.map_fn(self.propKernel_tf, self.z_planes_tf, dtype=tf.complex64))

        # else:
        #     self.prop_kern = np.asarray([propKernel(shape, pixel_size, wavelength, z_planes[zIdx], NA = NA) for zIdx in range(len(z_planes))])

        self.optimizer_z   = None
        self.optimizer_x   = None
        self.optimizer_pupil = None

    def set_z_pos(self, z_planes):
        self.z_planes_tf   = tf.Variable(np.array(z_planes).astype(np.float32))
        dim0           = self.shape[0] + 2*self.pad[1][0]
        dim1           = self.shape[1] + 2*self.pad[1][1]
        self.prop_kern = np.asarray([propKernel((dim0, dim1), self.pixel_size, self.wavelength, z_planes[zIdx],
                                                RI=self.RI_measure, NA = self.NA,fx_illu=self.fx_illu, fy_illu=self.fy_illu)
                                     for zIdx in range(len(z_planes))])

    def set_pupil(self, zernike_coef=None):
        coef = self.zernike_coeffs if zernike_coef is None else zernike_coef
        self.pupil = np.exp(1.0j * np.sum(self.zernike_bases * coef[:, np.newaxis, np.newaxis], axis=0)) \
                     * self.pupil_mask

    @tf.function
    def get_pupil_tf(self, zernike_coef_tf):
        pupil_tf = tf.exp(tf.complex(0.0, 1.0) * tf.reduce_sum(tf.constant(self.zernike_bases) *
                                                                tf.expand_dims(tf.expand_dims(tf.complex(zernike_coef_tf,0.0),axis=-1),
                                                                               axis=-1), axis=0)) * tf.constant(self.pupil_mask)
        return pupil_tf

    @tf.function
    def propKernel_tf(self, prop_distance):
        prop_kernel = tf.exp(tf.complex(0.0, 2.0) * tf.complex(tf.abs(prop_distance), 0.0) *
                                     tf.constant(np.pi * ((self.RI_measure / self.wavelength) ** 2 - (self.fxlin[naxis, :] - self.fx_illu) ** 2 -
                                                       (self.fylin[:, naxis] - self.fy_illu) ** 2) ** 0.5,dtype=tf.complex64))
        if tf.math.less(prop_distance, 0):
            prop_kernel = tf.math.conj(prop_kernel)
        return prop_kernel

    @tf.function
    def forward_tf(self, x, z_planes, zernike_coef, sel_vector):
        # x = tf.pad(x, [[self.pad[1][0], self.pad[1][0]], [self.pad[1][1], self.pad[1][1]]], 'CONSTANT')
        Fx  = tf.signal.fft2d(x)
        pupil = self.get_pupil_tf(zernike_coef)
        prop_kern_tf = tf.expand_dims(pupil,0) * tf.stack(tf.map_fn(self.propKernel_tf, z_planes, dtype=tf.complex64))
        Ax  = tf.signal.ifft2d(prop_kern_tf*tf.expand_dims(Fx,0))

        # if self.pad[0]:
        start0 = tf.constant(self.pad[1][0])
        start1 = tf.constant(self.pad[1][1])
        Ax = tf.slice(Ax, [0, start0, start1], [-1, tf.constant(len(self.ylin)), tf.constant(len(self.xlin))])
        # Ax     = Ax[:,start0:start0+len(self.ylin),start1:start1+len(self.xlin)]

        res     =  tf.abs(Ax)- tf.constant(self.amplitude,dtype=tf.float32)
        # if sel_vector is not None:
        res = tf.boolean_mask(res, sel_vector)
        funcVal = tf.norm(res)**2
        # funcVal = tf.reduce_mean(tf.abs(res+1e-7))
        # funcVal = tf.reduce_mean(res**2)
        return Ax, funcVal
        #     return funcVal

    @tf.function
    def _train_x(self, x_real, x_imag, z, zernike_coef, update_vector_, max_iter_):
        x = tf.complex(x_real, x_imag)
        _, e_0 = self.forward_tf(x, z, zernike_coef, update_vector_)

        for step in tf.range(tf.constant(max_iter_)):
            # update_vector_ran = tf.random.uniform(tf.constant([self.amplitude.shape[0]]), minval=0, maxval=2)
            # update_vector_ran = update_vector_ran * update_vector_
            with tf.GradientTape(persistent=True) as grad_tape:
                x = tf.complex(x_real, x_imag)
                _, e_0 = self.forward_tf(x, z, zernike_coef, update_vector_)

            grad_x_k_real = grad_tape.gradient(e_0, x_real)
            grad_x_k_imag = grad_tape.gradient(e_0, x_imag)
            self.optimizer_x.apply_gradients([(grad_x_k_real, x_real), (grad_x_k_imag, x_imag)])
            del grad_tape
            # tf.print('Updating field. step: ',step, ' error: ',e_0, end="\r")

            # x_k = tf.complex(x_real, x_imag)
            # _, e_1 = self.forward_tf(x_k, z, zernike_coef, update_vector_)
            # if e_1 > e_0:
            #     print("Updating x. meet convergence requirement at iteration %d!" % (step + 1))
            #     break

        return e_0

    def solve_tf(self, x_init=None, max_iter=100, lr=1e-2, tol=1e-5):
        if self.optimizer_x is None:
            self.optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr)

        if x_init is None:
            # x_init            = self.amplitude[self.amplitude.shape[0]//2]
            x_init            = np.ones_like(self.amplitude[0])
            # x_init = self.amplitude[0]

            if self.pad[0]:
                x_init = np.pad(x_init,((self.pad[1][0],),(self.pad[1][1],)),'constant',constant_values=(0.0,)).astype(np.complex64)

        update_vector = tf.ones((self.amplitude.shape[0]))
        error = []
        x_k_real = tf.Variable(x_init.real, trainable=True)
        x_k_imag = tf.Variable(x_init.imag, trainable=True)
        zernike_coef_tf = tf.Variable(self.zernike_coeffs, trainable=False)

        e_0 = self._train_x(x_k_real, x_k_imag, self.z_planes_tf, zernike_coef_tf,update_vector, max_iter)
        error.append(e_0.numpy())
        # error.append(e_1.numpy())

        # if step == 0 and error[-1] > error[-2]:
        #     print("Updating field. stepSize is too large!")
        #     return x_k.numpy(), error

        # if np.abs(error[-1] - error[-2]) / error[-1] < tol or error[-1] < 1e-20:
        #     print("Updating field. meet convergence requirement at iteration %d!" % (step + 1))
        #     return tf.complex(x_k_real, x_k_imag).numpy(), error

        print('Updating field. step: error: {:f}'.format(error[-1]),end="\r")
        return tf.complex(x_k_real, x_k_imag).numpy(), error

    @tf.function
    def _update_pupil(self, x_real, x_imag, z, zernike_coef_tf, update_vector_):
        x = tf.complex(x_real, x_imag)

        with tf.GradientTape(persistent=True) as grad_tape:
            _, e_0 = self.forward_tf(x, z, zernike_coef_tf, update_vector_)

        grad_coef = grad_tape.gradient(e_0, zernike_coef_tf)
        self.optimizer_pupil.apply_gradients([(grad_coef, zernike_coef_tf)])

        # self.get_pupil_tf(zernike_coef_tf)
        _, e_1 = self.forward_tf(x, z, zernike_coef_tf, update_vector_)
        del grad_tape
        return e_0, e_1

    @tf.function
    def _train_z(self,x_real, x_imag,z,zernike_coef, update_vector_, w_tv1, w_tv2):
        x = tf.complex(x_real, x_imag)
        with tf.GradientTape(persistent=True) as grad_tape:
            _, e_0 = self.forward_tf(x, z, zernike_coef, update_vector_)
            error = e_0 + w_tv2 * tf.reduce_sum(tf.abs((z[1:] - z[:-1])[1:] - (z[1:] - z[:-1])[:-1]))
            # w_tv1 * tf.reduce_sum(tf.abs(z[1:] - z[:-1]))
        grad_z = grad_tape.gradient(error, z)
        self.optimizer_z.apply_gradients([(grad_z, z)])
        _, e_1 = self.forward_tf(x, z, zernike_coef, update_vector_)
        del grad_tape
        return e_0, e_1

    def solve_z(self, x, z_init, update_vector=None, max_iter=100, lr=1e-2, tol=1e-6):
        error = []
        x_k_real = tf.Variable(x.real, trainable=False)
        x_k_imag = tf.Variable(x.imag, trainable=False)
        zernike_coef_tf = tf.Variable(self.zernike_coeffs, trainable=False)
        z_planes_tf = tf.Variable(np.array(z_init).astype(np.float32), trainable=True)
        z_log = [z_init]

        if self.optimizer_z is None:
            self.optimizer_z = tf.keras.optimizers.Adam(learning_rate=lr)

        if update_vector is None:
            update_vector = tf.ones((self.amplitude.shape[0]))

        for step in range(max_iter):
            e_0, e_1 = self._train_z(x_k_real, x_k_imag, z_planes_tf,zernike_coef_tf,update_vector)
            error.append(e_0.numpy())
            # self.prop_kern_tf = self.calculate_prop_kern_tf(z_planes_tf)
            error.append(e_1.numpy())

            # if step == 0 and error[-1] > error[-2]:
            #     print("Updating z. stepSize is too large!")
            #     return z_log, error

            if np.abs(error[-1] - error[-2]) / error[-1] < tol or error[-1] < 1e-20:
                print("Updating z. meet convergence requirement at iteration %d!" % (step + 1))
                return z_log, error

            print('Updating z. step: {}, error: {:f}'.format(step, error[-1]),end="\r")
            # z_log.append(np.array([z.numpy() for z in z_planes_tf]))

        z_log.append(z_planes_tf.numpy())

        # z_mean = np.mean(z_log[-1])
        # self.z_planes_tf = z_planes_tf -  z_mean
        self.set_z_pos(z_planes_tf.numpy())
        # self.prop_kern_tf = tf.stack(tf.map_fn(self.propKernel_tf, self.z_planes_tf, dtype=tf.complex64))

        return z_log, error

    def pupil_recovery(self, x, z, update_vector=None, max_iter=100, lr=1e-2, tol=1e-6):
        x_k_real = tf.Variable(x.real, trainable=False)
        x_k_imag = tf.Variable(x.imag, trainable=False)
        z_planes_tf = tf.Variable(np.array(z).astype(np.float32), trainable=False)
        zernike_coef_tf = tf.Variable(self.zernike_coeffs, trainable=True)
        zernike_log = [self.zernike_coeffs]
        error = []

        if self.optimizer_pupil is None:
            self.optimizer_pupil = tf.keras.optimizers.Adam(learning_rate=lr)
        if update_vector is None:
            update_vector = tf.ones((self.amplitude.shape[0]))

        for step in range(max_iter):
            e_0, e_1 = self._update_pupil(x_k_real, x_k_imag, z_planes_tf, zernike_coef_tf, update_vector)
            error.append(e_0.numpy())
            error.append(e_1.numpy())

            # if np.abs(error[-1] - error[-2]) / error[-1] < tol or error[-1] < 1e-20:
            #     print("Updating z. meet convergence requirement at iteration %d!" % (step + 1))
            #     self.set_pupil(zernike_log[-1])
            #     return zernike_log, error

            zernike_log.append(zernike_coef_tf.numpy())
            print('Updating pupil. step: {}, error: {:f}'.format(step, error[-2]), end="\r")

        self.set_pupil(zernike_log[-1])
        return zernike_log, error

    def joint_solve_xz(self, z_init, x_init=None, lr_x=2e-2, lr_z=2e-2, lr_pupil=1e-2, iterations=50, max_iter_x=50,
                       max_iter_z=10, max_iter_pupil=0):
        z_log, x_log, zernike_log = [], [], []
        err_x_log, err_z_log, err_pupil_log = [], [], []
        z_center_ind = np.argmin(np.abs(z_init))
        if x_init is None:
            x_init            = self.amplitude[self.amplitude.shape[0]//2]
            if self.pad[0]:
                x_init = np.pad(x_init,((self.pad[1][0],),(self.pad[1][1],)),'constant',constant_values=(0.0,)).astype(np.complex64)

        z_log.append(np.array(z_init).astype(np.float32))

        self.optimizer_z = tf.keras.optimizers.Adam(learning_rate=lr_z)
        self.optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)
        self.optimizer_pupil = tf.keras.optimizers.Adam(learning_rate=lr_pupil)

        x_k_real = tf.Variable(x_init.real, trainable=True)
        x_k_imag = tf.Variable(x_init.imag, trainable=True)
        z_planes_tf = tf.Variable(np.array(z_init).astype(np.float32), trainable=True)
        zernike_coef_tf = tf.Variable(self.zernike_coeffs, trainable=True)
        update_vector = tf.ones((self.amplitude.shape[0]))
        # update_vector = tf.random.uniform((self.amplitude.shape[0]), minval=0, maxval=2)

        for s in range(iterations):
            # update x
            tf.compat.v1.variables_initializer(self.optimizer_x.variables())
            e_0 = self._train_x(x_k_real, x_k_imag, z_planes_tf, zernike_coef_tf, update_vector, max_iter_x)
            err_x_log.append(e_0.numpy())
            x_log.append(tf.complex(x_k_real, x_k_imag).numpy())

            # update z
            tf.compat.v1.variables_initializer(self.optimizer_z.variables())
            error = [0.0]
            for step in range(max_iter_z):
                e_0, e_1 = self._train_z(x_k_real, x_k_imag, z_planes_tf, zernike_coef_tf,update_vector, 5.0, 0) # 2e-1
                error.append(e_0.numpy())

                if (np.abs(e_1 - e_0) / e_1) < 1e-6 or error[-1] < 1e-20:
                    print("Updating z. meet convergence requirement at iteration %d!" % (step + 1))
                    break

                print('Updating z. step: {}, error: {:f}'.format(step, error[-1]),end="\r")

            z_pos_t = z_planes_tf.numpy()
            z_pos_nondecr = np.ones_like(z_log[-1]) * z_init[0]
            z_pos_nondecr[1:] += np.cumsum(np.maximum(z_pos_t[1::] - z_pos_t[:-1], 0.0))
            z_pos_nondecr = z_pos_nondecr - (z_pos_nondecr[z_center_ind] - z_init[z_center_ind])

            self.set_z_pos(z_pos_nondecr)
            z_log.append(z_pos_nondecr)
            err_z_log.append(error[-1])

            # embedded pupil recovery
            # tf.compat.v1.variables_initializer(self.optimizer_pupil.variables())
            error = [0.0]
            for step in range(max_iter_pupil):
                e_0, e_1 = self._update_pupil(x_k_real, x_k_imag, z_planes_tf, zernike_coef_tf, update_vector)
                error.append(e_0.numpy())
                error.append(e_1.numpy())

                print('Updating pupil. step: {}, error: {:f}'.format(step, error[-1]),end="\r")

            zernike_log.append(zernike_coef_tf.numpy())
            err_pupil_log.append(error[-1])

        return x_log, z_log, zernike_log, err_x_log, err_z_log, err_pupil_log


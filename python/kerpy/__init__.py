from abc import abstractmethod
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import (arange, cos, dot, exp, fill_diagonal, mean,
                   shape, sin, sqrt, zeros)
from numpy.random import permutation, randn
from scipy import linalg
from scipy.linalg import sqrtm, inv
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.stats import norm as normaldist
import time


def hsic(x, y):
    if x.shape[0] > 222:
        inds = np.random.choice(x.shape[0],
                                size=222,
                                replace=False)
        X = x[inds, :]
        Y = y[inds, :]
    else:
        X = x
        Y = y
    hsic = HSICPermutationTestObject(X.shape[0],
                                     kernelX=GaussianKernel(),
                                     kernelY=GaussianKernel())
    return hsic.compute_pvalue(X, Y)


class TestObject(object):

    def __init__(self, test_type, streaming=False, freeze_data=False):
        self.test_type = test_type
        self.streaming = streaming
        self.freeze_data = freeze_data
        if self.freeze_data:
            self.generate_data()
            assert not self.streaming

    @abstractmethod
    def compute_Zscore(self):
        raise NotImplementedError

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError

    def compute_pvalue(self):
        Z_score = self.compute_Zscore()
        pvalue = normaldist.sf(Z_score)
        return pvalue

    def perform_test(self, alpha):
        pvalue = self.compute_pvalue()
        return pvalue < alpha


class HSICTestObject(TestObject):
    def __init__(self,
                 num_samples,
                 data_generator=None,
                 kernelX=None,
                 kernelY=None,
                 kernelZ=None,
                 kernelX_use_median=False,
                 kernelY_use_median=False,
                 kernelZ_use_median=False,
                 rff=False,
                 num_rfx=None,
                 num_rfy=None,
                 induce_set=False,
                 num_inducex=None,
                 num_inducey=None,
                 streaming=False,
                 freeze_data=False):
        TestObject.__init__(self,
                            self.__class__.__name__,
                            streaming=streaming,
                            freeze_data=freeze_data)
        # We have same number of samples from X and Y in independence testing
        self.num_samples = num_samples
        self.data_generator = data_generator
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        # indicate if median heuristic for Gaussian Kernel should be used
        self.kernelX_use_median = kernelX_use_median
        self.kernelY_use_median = kernelY_use_median
        self.kernelZ_use_median = kernelZ_use_median
        self.rff = rff
        self.num_rfx = num_rfx
        self.num_rfy = num_rfy
        self.induce_set = induce_set
        self.num_inducex = num_inducex
        self.num_inducey = num_inducey
        if self.rff | self.induce_set:
            self.HSICmethod = self.HSIC_with_shuffles_rff
        else:
            self.HSICmethod = self.HSIC_with_shuffles

    def generate_data(self, isConditionalTesting=False):
        if not isConditionalTesting:
            self.data_x, self.data_y = self.data_generator(self.num_samples)
            return self.data_x, self.data_y
        else:
            self.data_x, self.data_y, self.data_z = self.data_generator(
                self.num_samples)
            return self.data_x, self.data_y, self.data_z

    @staticmethod
    def HSIC_U_statistic(Kx, Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx, 0.)
        fill_diagonal(Ky, 0.)
        K = np.dot(Kx, Ky)
        first_term = np.trace(K)/float(m*(m-3.))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term

    @staticmethod
    def HSIC_V_statistic(Kx, Ky):
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        return np.sum(Kxc*Kyc)

    @staticmethod
    def HSIC_V_statistic_rff(phix, phiy):
        m = shape(phix)[0]
        phix_c = phix-mean(phix, axis=0)
        phiy_c = phiy-mean(phiy, axis=0)
        featCov = (phix_c.T).dot(phiy_c)/float(m)
        return np.linalg.norm(featCov)**2

    # generalise distance correlation ---- a kernel interpretation
    @staticmethod
    def dCor_HSIC_statistic(Kx, Ky, unbiased=False):
        if unbiased:
            first_term = HSICTestObject.HSIC_U_statistic(Kx, Ky)
            second_term = HSICTestObject.HSIC_U_statistic(Kx, Kx) \
                * HSICTestObject.HSIC_U_statistic(Ky, Ky)
            dCor = first_term/float(sqrt(second_term))
        else:
            first_term = HSICTestObject.HSIC_V_statistic(Kx, Ky)
            second_term = HSICTestObject.HSIC_V_statistic(Kx, Kx) \
                * HSICTestObject.HSIC_V_statistic(Ky, Ky)
            dCor = first_term/float(sqrt(second_term))
        return dCor

    # approximated dCor using rff/Nystrom
    @staticmethod
    def dCor_HSIC_statistic_rff(phix, phiy):
        first_term = HSICTestObject.HSIC_V_statistic_rff(phix, phiy)
        second_term = HSICTestObject.HSIC_V_statistic_rff(phix, phix) \
            * HSICTestObject.HSIC_V_statistic_rff(phiy, phiy)
        approx_dCor = first_term/float(sqrt(second_term))
        return approx_dCor

    def SubdCor_HSIC_statistic(self, data_x=None, data_y=None, unbiased=True):
        if data_x is None:
            data_x = self.data_x
        if data_y is None:
            data_y = self.data_y
        dx = shape(data_x)[1]
        stats_value = zeros(dx)
        for dd in range(dx):
            Kx, Ky = self.compute_kernel_matrix_on_data(
                data_x[:, [dd]], data_y)
            stats_value[dd] = HSICTestObject.dCor_HSIC_statistic(
                Kx, Ky, unbiased)
        SubdCor = sum(stats_value)/float(dx)
        return SubdCor

    def SubHSIC_statistic(self, data_x=None, data_y=None, unbiased=True):
        if data_x is None:
            data_x = self.data_x
        if data_y is None:
            data_y = self.data_y
        dx = shape(data_x)[1]
        stats_value = zeros(dx)
        for dd in range(dx):
            Kx, Ky = self.compute_kernel_matrix_on_data(
                data_x[:, [dd]], data_y)
            if unbiased:
                stats_value[dd] = HSICTestObject.HSIC_U_statistic(Kx, Ky)
            else:
                stats_value[dd] = HSICTestObject.HSIC_V_statistic(Kx, Ky)
        SubHSIC = sum(stats_value)/float(dx)
        return SubHSIC

    def HSIC_with_shuffles(self,
                           data_x=None,
                           data_y=None,
                           unbiased=True,
                           num_shuffles=0,
                           estimate_nullvar=False,
                           isBlockHSIC=False):
        start = time.time()
        if data_x is None:
            data_x = self.data_x
        if data_y is None:
            data_y = self.data_y
        time_passed = time.time() - start
        if isBlockHSIC:
            Kx, Ky = self.compute_kernel_matrix_on_dataB(data_x, data_y)
        else:
            Kx, Ky = self.compute_kernel_matrix_on_data(data_x, data_y)
        ny = shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic(Kx, Ky)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic(Kx, Ky)
        null_samples = zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            Kpp = Ky[pp, :][:, pp]
            if unbiased:
                null_samples[jj] = HSICTestObject.HSIC_U_statistic(Kx, Kpp)
            else:
                null_samples[jj] = HSICTestObject.HSIC_V_statistic(Kx, Kpp)
        if estimate_nullvar:
            nullvarx, nullvary = \
                self.unbiased_HSnorm_estimate_of_centred_operator(Kx, Ky)
            nullvarx = 2. * nullvarx
            nullvary = 2. * nullvary
        else:
            nullvarx, nullvary = None, None
        return (test_statistic,
                null_samples,
                nullvarx,
                nullvary,
                Kx,
                Ky,
                time_passed)

    def HSIC_with_shuffles_rff(self,
                               data_x=None,
                               data_y=None,
                               unbiased=True,
                               num_shuffles=0,
                               estimate_nullvar=False):
        start = time.clock()
        if data_x is None:
            data_x = self.data_x
        if data_y is None:
            data_y = self.data_y
        time_passed = time.clock()-start
        if self.rff:
            phix, phiy = self.compute_rff_on_data(data_x, data_y)
        else:
            phix, phiy = self.compute_induced_kernel_matrix_on_data(
                data_x, data_y)
        ny = shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic_rff(phix, phiy)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic_rff(phix, phiy)
        null_samples = zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            if unbiased:
                null_samples[jj] = HSICTestObject.HSIC_U_statistic_rff(
                    phix, phiy[pp])
            else:
                null_samples[jj] = HSICTestObject.HSIC_V_statistic_rff(
                    phix, phiy[pp])
        if estimate_nullvar:
            raise NotImplementedError()
        else:
            nullvarx, nullvary = None, None
        return (test_statistic,
                null_samples,
                nullvarx,
                nullvary,
                phix,
                phiy,
                time_passed)

    def get_spectrum_on_data(self, Mx, My):
        '''Mx and My are Kx Ky when rff =False
           Mx and My are phix, phiy when rff =True'''
        if self.rff | self.induce_set:
            Cx = np.cov(Mx.T)
            Cy = np.cov(My.T)
            lambdax = np.linalg.eigvalsh(Cx)
            lambday = np.linalg.eigvalsh(Cy)
        else:
            Kxc = Kernel.center_kernel_matrix(Mx)
            Kyc = Kernel.center_kernel_matrix(My)
            lambdax = np.linalg.eigvalsh(Kxc)
            lambday = np.linalg.eigvalsh(Kyc)
        return lambdax, lambday

    @abstractmethod
    def compute_kernel_matrix_on_data(self, data_x, data_y):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kx = self.kernelX.kernel(data_x)
        Ky = self.kernelY.kernel(data_y)
        return Kx, Ky

    @abstractmethod
    def compute_kernel_matrix_on_dataB(self, data_x, data_y):
        Kx = self.kernelX.kernel(data_x)
        Ky = self.kernelY.kernel(data_y)
        return Kx, Ky

    @abstractmethod
    def compute_kernel_matrix_on_data_CI(self, data_x, data_y, data_z):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        if self.kernelZ_use_median:
            sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
            self.kernelZ.set_width(float(sigmaz))
        Kx = self.kernelX.kernel(data_x)
        Ky = self.kernelY.kernel(data_y)
        Kz = self.kernelZ.kernel(data_z)
        return Kx, Ky, Kz

    def unbiased_HSnorm_estimate_of_centred_operator(self, Kx, Ky):
        '''returns an unbiased estimate of 2*Sum_p Sum_q lambda^2_p theta^2_q
        where lambda and theta are the eigenvalues
        of the centered matrices for X and Y respectively'''
        varx = HSICTestObject.HSIC_U_statistic(Kx, Kx)
        vary = HSICTestObject.HSIC_U_statistic(Ky, Ky)
        return varx, vary

    @abstractmethod
    def compute_rff_on_data(self, data_x, data_y):
        self.kernelX.rff_generate(self.num_rfx, dim=shape(data_x)[1])
        self.kernelY.rff_generate(self.num_rfy, dim=shape(data_y)[1])
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        phix = self.kernelX.rff_expand(data_x)
        phiy = self.kernelY.rff_expand(data_y)
        return phix, phiy

    @abstractmethod
    def compute_induced_kernel_matrix_on_data(self, data_x, data_y):
        '''Z follows the same distribution as X; W follows that of Y.
        The current data generating methods we use
        generate X and Y at the same time. '''
        size_induced_set = max(self.num_inducex, self.num_inducey)
        if self.data_generator is None:
            subsample_idx = np.random.randint(
                self.num_samples, size=size_induced_set)
            self.data_z = data_x[subsample_idx, :]
            self.data_w = data_y[subsample_idx, :]
        else:
            self.data_z, self.data_w = self.data_generator(size_induced_set)
            self.data_z[[range(self.num_inducex)], :]
            self.data_w[[range(self.num_inducey)], :]
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kxz = self.kernelX.kernel(data_x, self.data_z)
        Kzz = self.kernelX.kernel(self.data_z)
        # R = inv(sqrtm(Kzz))
        R = inv(sqrtm(Kzz + np.eye(np.shape(Kzz)[0])*10**(-6)))
        phix = Kxz.dot(R)
        Kyw = self.kernelY.kernel(data_y, self.data_w)
        Kww = self.kernelY.kernel(self.data_w)
        # S = inv(sqrtm(Kww))
        S = inv(sqrtm(Kww + np.eye(np.shape(Kww)[0])*10**(-6)))
        phiy = Kyw.dot(S)
        return phix, phiy

    def compute_pvalue(self, data_x=None, data_y=None):
        pvalue, _ = self.compute_pvalue_with_time_tracking(data_x, data_y)
        return pvalue


class HSICPermutationTestObject(HSICTestObject):

    def __init__(self,
                 num_samples,
                 data_generator=None,
                 kernelX=None,
                 kernelY=None,
                 kernelX_use_median=False,
                 kernelY_use_median=False,
                 num_rfx=None,
                 num_rfy=None,
                 rff=False,
                 induce_set=False,
                 num_inducex=None,
                 num_inducey=None,
                 num_shuffles=500,
                 unbiased=True):
        HSICTestObject.__init__(self,
                                num_samples,
                                data_generator=data_generator,
                                kernelX=kernelX,
                                kernelY=kernelY,
                                kernelX_use_median=kernelX_use_median,
                                kernelY_use_median=kernelY_use_median,
                                num_rfx=num_rfx,
                                num_rfy=num_rfy,
                                rff=rff,
                                induce_set=induce_set,
                                num_inducex=num_inducex,
                                num_inducey=num_inducey)
        self.num_shuffles = num_shuffles
        self.unbiased = unbiased

    def compute_pvalue_with_time_tracking(self, data_x=None, data_y=None):
        if data_x is None and data_y is None:
            if not self.streaming and not self.freeze_data:
                start = time.clock()
                self.generate_data()
                data_generating_time = time.clock()-start
                data_x = self.data_x
                data_y = self.data_y
            else:
                data_generating_time = 0.
        else:
            data_generating_time = 0.
        hsic_statistic, null_samples = self.HSICmethod(
            unbiased=self.unbiased,
            num_shuffles=self.num_shuffles,
            data_x=data_x,
            data_y=data_y)[:2]
        pvalue = (1 + sum(null_samples > hsic_statistic)) \
            / float(1 + self.num_shuffles)

        return pvalue, data_generating_time


class GenericTests():
    @staticmethod
    def check_type(varvalue, varname, vartype, required_shapelen=None):
        if not type(varvalue) is vartype:
            raise TypeError(
                "Variable " + varname
                + " must be of type " + vartype.__name__ +
                ". Given is " + str(type(varvalue)))
        if required_shapelen is not None:
            if not len(varvalue.shape) is required_shapelen:
                raise ValueError(
                    "Variable " + varname
                    + " must be " + str(required_shapelen) + "-dimensional")
        return 0


class Kernel(object):
    def __init__(self):
        self.rff_num = None
        self.rff_freq = None
        pass

    def __str__(self):
        s = ""
        return s

    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()

    @abstractmethod
    def set_kerpar(self, kerpar):
        self.set_width(kerpar)

    @abstractmethod
    def set_width(self, width):
        if hasattr(self, 'width'):
            if self.rff_freq is not None:
                self.rff_freq = self.unit_rff_freq / width
            self.width = width
        else:
            raise ValueError("Senseless: kernel has no 'width' attribute!")

    @abstractmethod
    def rff_generate(self, m, dim=1):
        raise NotImplementedError()

    @abstractmethod
    def rff_expand(self, X):
        if self.rff_freq is None:
            raise ValueError(
                "rff_freq has not been set. use rff_generate first")
        """
        Computes the random Fourier features for the input dataset X
        for a set of frequencies in rff_freq.
        This set of frequencies has to be precomputed
        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        """
        GenericTests.check_type(X, 'X', np.ndarray)
        xdotw = dot(X, (self.rff_freq).T)
        return sqrt(2. / self.rff_num) * np.concatenate(
            (cos(xdotw), sin(xdotw)), axis=1)

    @abstractmethod
    def gradient(self, x, Y):

        # ensure this in every implementation
        assert(len(shape(x)) == 1)
        assert(len(shape(Y)) == 2)
        assert(len(x) == shape(Y)[1])

        raise NotImplementedError()

    @staticmethod
    def centering_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return np.eye(n) - 1.0 / n

    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n
        and returns HKH
        """
        n = shape(K)[0]
        H = np.eye(n) - 1.0 / n
        return 1.0 / n * H.dot(K.dot(H))

    @abstractmethod
    def svc(self, X, y, lmbda=1.0, Xtst=None, ytst=None):
        from sklearn import svm
        svc = svm.SVC(kernel=self.kernel, C=lmbda)
        svc.fit(X, y)
        if Xtst is None:
            return svc
        else:
            ypre = svc.predict(Xtst)
            if ytst is None:
                return svc, ypre
            else:
                return svc, ypre, 1-svc.score(Xtst, ytst)

    @abstractmethod
    def svc_rff(self, X, y, lmbda=1.0, Xtst=None, ytst=None):
        from sklearn import svm
        phi = self.rff_expand(X)
        svc = svm.LinearSVC(C=lmbda, dual=True)
        svc.fit(phi, y)
        if Xtst is None:
            return svc
        else:
            phitst = self.rff_expand(Xtst)
            ypre = svc.predict(phitst)
            if ytst is None:
                return svc, ypre
            else:
                return svc, ypre, 1-svc.score(phitst, ytst)

    @abstractmethod
    def ridge_regress(self, X, y, lmbda=0.01, Xtst=None, ytst=None):
        K = self.kernel(X)
        n = shape(K)[0]
        aa = linalg.solve(K + lmbda * np.eye(n), y)
        if Xtst is None:
            return aa
        else:
            ypre = dot(aa.T, self.kernel(X, Xtst)).T
            if ytst is None:
                return aa, ypre
            else:
                return aa, ypre, (linalg.norm(ytst-ypre)**2)/np.shape(ytst)[0]

    @abstractmethod
    def ridge_regress_rff(self, X, y, lmbda=0.01, Xtst=None, ytst=None):
        phi = self.rff_expand(X)
        bb = linalg.solve(
            dot(phi.T, phi)+lmbda*np.eye(self.rff_num), dot(phi.T, y))
        if Xtst is None:
            return bb
        else:
            phitst = self.rff_expand(Xtst)
            ypre = dot(phitst, bb)
            if ytst is None:
                return bb, ypre
            else:
                return bb, ypre, (linalg.norm(ytst-ypre)**2)/np.shape(ytst)[0]

    @abstractmethod
    def xvalidate(self,
                  X,
                  y,
                  method='ridge_regress',
                  regpar_grid=(1+arange(25))/200.0,
                  kerpar_grid=exp(-13+arange(25)),
                  numFolds=10,
                  verbose=False,
                  visualise=False):
        from sklearn import cross_validation
        which_method = getattr(self, method)
        n = len(X)
        kf = cross_validation.KFold(n, n_folds=numFolds)
        xvalerr = zeros((len(regpar_grid), len(kerpar_grid)))
        width_idx = 0
        for width in kerpar_grid:
            try:
                self.set_kerpar(width)
            except ValueError:
                xvalerr[:, width_idx] = np.inf
                width_idx += 1
                continue
            else:
                lmbda_idx = 0
                for lmbda in regpar_grid:
                    fold = 0
                    prederr = zeros(numFolds)
                    for train_index, test_index in kf:
                        if type(X) == list:
                            X_train = [X[i] for i in train_index]
                            X_test = [X[i] for i in test_index]
                        else:
                            X_train, X_test = X[train_index], X[test_index]
                        if type(y) == list:
                            y_train = [y[i] for i in train_index]
                            y_test = [y[i] for i in test_index]
                        else:
                            y_train, y_test = y[train_index], y[test_index]
                        _, _, prederr[fold] = which_method(X_train,
                                                           y_train,
                                                           lmbda=lmbda,
                                                           Xtst=X_test,
                                                           ytst=y_test)
                        fold += 1
                    xvalerr[lmbda_idx, width_idx] = mean(prederr)
                    lmbda_idx += 1
                width_idx += 1
        min_idx = np.unravel_index(np.argmin(xvalerr), shape(xvalerr))
        if visualise:
            plt.imshow(xvalerr,
                       interpolation='none',
                       origin='lower',
                       cmap=cm.pink)
            plt.colorbar()
            plt.title("cross-validated loss")
            plt.ylabel("regularisation parameter")
            plt.xlabel("kernel parameter")
            plt.show()
        return regpar_grid[min_idx[0]], kerpar_grid[min_idx[1]]

    @abstractmethod
    def estimateMMD(self, sample1, sample2, unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1)
        K22 = self.kernel(sample2)
        K12 = self.kernel(sample1, sample2)
        if unbiased:
            fill_diagonal(K11, 0.0)
            fill_diagonal(K22, 0.0)
            n = float(shape(K11)[0])
            m = float(shape(K22)[0])
            return sum(sum(K11))/(pow(n, 2)-n) \
                + sum(sum(K22))/(pow(m, 2)-m) - 2*mean(K12[:])
        else:
            return mean(K11[:])+mean(K22[:])-2*mean(K12[:])

    @abstractmethod
    def estimateMMD_rff(self, sample1, sample2, unbiased=False):
        phi1 = self.rff_expand(sample1)
        phi2 = self.rff_expand(sample2)
        featuremean1 = mean(phi1, axis=0)
        featuremean2 = mean(phi2, axis=0)
        if unbiased:
            nx = shape(phi1)[0]
            ny = shape(phi2)[0]
            first_term = nx/(nx-1.0)*(dot(featuremean1, featuremean1)
                                      - mean(linalg.norm(phi1, axis=1)**2)/nx)
            second_term = ny/(ny-1.0)*(dot(featuremean2, featuremean2)
                                       - mean(linalg.norm(phi2, axis=1)**2)/ny)
            third_term = -2*dot(featuremean1, featuremean2)
            return first_term+second_term+third_term
        else:
            return linalg.norm(featuremean1-featuremean2)**2


class GaussianKernel(Kernel):
    def __init__(self, sigma=1.0, is_sparse=False):
        Kernel.__init__(self)
        self.width = sigma
        self.is_sparse = is_sparse

    def __str__(self):
        s = self.__class__.__name__ + "["
        s += "width=" + str(self.width)
        s += "]"
        return s

    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)

        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples,
            can be None in which case its replaced by X
        """
        if self.is_sparse:
            X = X.todense()
            Y = Y.todense()
        GenericTests.check_type(X, 'X', np.ndarray)
        assert(len(shape(X)) == 2)

        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            GenericTests.check_type(Y, 'Y', np.ndarray)
            assert(len(shape(Y)) == 2)
            assert(shape(X)[1] == shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')

        K = exp(-0.5 * (sq_dists) / self.width ** 2)
        return K

    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel
        wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        """
        if self.is_sparse:
            x = x.todense()
            Y = Y.todense()
        assert(len(shape(x)) == 1)
        assert(len(shape(Y)) == 2)
        assert(len(x) == shape(Y)[1])

        x_2d = np.reshape(x, (1, len(x)))
        k = self.kernel(x_2d, Y)
        differences = Y - x
        G = (1.0 / self.width ** 2) * (k.T * differences)
        return G

    def rff_generate(self, m, dim=1):
        self.rff_num = m
        self.unit_rff_freq = randn(int(m/2), dim)
        self.rff_freq = self.unit_rff_freq/self.width

    @staticmethod
    def get_sigma_median_heuristic(X, is_sparse=False):
        if is_sparse:
            X = X.todense()
        n = shape(X)[0]
        if n > 1000:
            X = X[permutation(n)[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = np.median(dists[dists > 0])
        sigma = median_dist/sqrt(2.)
        return sigma

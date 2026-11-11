import numpy as np

class MathModel:
    """
    Abstract class for handling any type of mathematical equation we encounter
    """
    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.params = []
    def extract_params(self,):
        for i, param in enumerate(self.kwargs.keys()):
            self.params.append(self.kwargs[param])
        # There should be atleast 4 parameters and other parameters will be discarded
        assert len(self.params) >= 4

    def model(self,X):
        pass

class ExponentialModel(MathModel):
    """
    An exponential function should be of 
        a_0 * exp {a_1 + a_2 * x} + a_3
    """
    def __init__(self,*args, **kwargs):
        super().__init__(args, kwargs)
        self.extract_params()
    
    def model(self, X):
        return self.params[0] * np.exp(self.params[1]+self.params[2] * X) + self.params[3]

class LinearModel(MathModel):
    """
    An linear function should be of,
        a_0 + a_1 * x
    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.extract_params()

    def model(self, X):
        # Make sure X should be 1D numby array or a simple list
        if isinstance(X, list):
            return [self.params[0]+self.params[1]*x for x in X]
        return self.params[0] + self.params[1] * X
    
class LinExponentialModel_2D(MathModel):
    """
    A linear + exponential model looks like,
        a_0 + a_1 * x_1 + a_2 * exp{a_3 + a_4 * x_2}
    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, **kwargs)
        self.extract_params()
    
    def model(self, X, **kwargs):
        is_clip = kwargs.get("clip", False)
        x_1, x_2 = X
        if isinstance(x_1, np.ndarray) and isinstance(x_2, np.ndarray):
            if is_clip:
                # this is one is required for handling numpy dtype = np.float128 or so
                # in mac float128 is not supported
                exp_input = np.clip(x_2 * self.params[2], None, 600)
            else:
                exp_input = x_2 * self.params[2]
            exp_term = np.exp(exp_input)
            return np.abs
        else:
            raise NotImplementedError(f"types other tha numpy are not implemennted yet")

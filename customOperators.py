import odl
import numpy as np

class RadonTimeAdjoint(odl.Operator):
    def __init__(self, A, range):
        #self.domain = domain
        #self.range = range
        self.diagop = A
        super().__init__(self.diagop.domain, range, True)

    
    def _call(self,x):
        #This time input is in right format for diagonal operator but must be reshaped to correct output
        product_space_result = self.diagop(x)
        data_in_operator_range = self.range.element(np.transpose(np.array([part.data for part in product_space_result.parts]),(1,2,0)))
        return data_in_operator_range
        
    
class RadonTime(odl.Operator):
    def __init__(self, *A, domain):
        #self.domain = domain
        #self.range = range
        self.diagop = odl.DiagonalOperator(*A)
        super().__init__(domain, self.diagop.range, True)

    
    def _call(self,x):
        #x is an nd image with last dimension being time. Reshape into product space where dimension of time is the power
        data_in_diag_domain = self.diagop.domain.element(np.transpose(x.data, (2,0,1)))
        product_space_result = self.diagop(data_in_diag_domain)
        return product_space_result
        
    
    @property
    def adjoint(self):
        return RadonTimeAdjoint(self.diagop.adjoint, self.domain)

from tensorflow.keras.layers import Dropout, AlphaDropout

class SpecDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.training=False
    
    def call(self, inputs):
        return super().call(inputs,self.training)


class SpecAlphaDropout(AlphaDropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.training=False
    
    def call(self, inputs):
        return super().call(inputs,self.training)
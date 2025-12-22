from mempyfit import Parameters

parameters = Parameters({
            'r' : {'value' : 0.2, 'free' : True, 'unit' : 'd^-1', 'label' : 'growth rate'}, 
            'K' : {'value' : 1.0, 'free' : True, 'unit' : '-', 'label' : 'carrying capacity'},
        })
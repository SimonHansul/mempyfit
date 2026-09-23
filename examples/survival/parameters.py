from mempyfit import Parameters 

parameters = Parameters({
    'h_b' : {'value' : 0.001,   'free' : True,  'unit' : '1/d', 'label' : 'background hazard rate'},
    'k_d' : {'value' : 0.1,   'free' : True,  'unit' : '1/d', 'label' : 'dominant rate constant'},
    'b'   : {'value' : 0.001,   'free' : True,  'unit' : 'nM Ni$^{2+}$ d$^{-1}$', 'label' : 'killing rate'}, 
    'z'   : {'value' : 300,   'free' : True,  'unit' : 'nM Ni$^{2+}$', 'label' : 'threshold'},
})

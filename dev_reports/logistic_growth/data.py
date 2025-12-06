from mempyfit import *
data = Dataset()

data.add(
    name = 't-OD', 
    value = np.array([
        [ -0.05911330049261121,0.10723981900452495],
        [2.0886699507389164,0.10723981900452495],
        [3.9211822660098505,0.12760180995475118],
        [6.029556650246304,0.15656108597285068],
        [8.019704433497536,0.1773755656108597],
        [10.049261083743838,0.19140271493212674],
        [12.039408866995073,0.2],
        [14.088669950738913,0.20769230769230773],
        [16.039408866995075,0.21176470588235297],
        [18.00985221674877,0.21312217194570138]
    ]), 
    units = ['d', '-'], 
    labels = ['time', 'optical density'],
    title = 'Optical density of Bd over time',
    bibkey = 'voylesDiversityGrowthPatterns2017' #  we can add a bibkey from Zotero to document literature sources 
)

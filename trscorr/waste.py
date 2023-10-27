def quartic_p(x):
    return numpy.exp(x**4)

def quarticp_hs_transform(x, nsamples = nsamples):
    k = numpy.random.randn(nsamples)
    fx = numpy.exp(numpy.sqrt(2)*k*x**2) 
    return fx.mean(), fx.std()

def quarticp_double_hs_transform(x, nsamples = nsamples):
    k0 = numpy.random.randn(2*nsamples)
    k = k0[:nsamples]
    l = k0[nsamples+1:]
    b = 2. * numpy.sqrt(2)*k 
    #print(numpy.sqrt(b))
    fx = numpy.exp(-numpy.outer*np.emath.sqrt(b)*l*x).ravel()
    # return fx.mean(), fx.std()/numpy.sqrt(nsamples)
    return fx.mean(), fx.std()
class hay:

    def __init__(self, q10):
        self.q10 = q10
    
    def na_steady_dend(self,v):
        
        if v == -32:
            v += .0001

        alpha = (0.182 * (v- -32))/(1-(np.exp(-(v - -32)/6)))
        beta = (0.124 * (-v -32))/(1-(np.exp(-(-v -32)/6)))
        sum = alpha + beta
        mtau = (1/sum)/self.q10
        minf = alpha/sum

        if v == -60:
            v += .0001

        alpha = (-0.015 * (v- -60))/(1-(np.exp((v- -60)/6)))
        beta = (-0.015 * (-v -60))/(1-(np.exp((-v -60)/6)))
        sum = alpha + beta
        htau = (1/sum)/self.q10
        hinf = alpha/sum

        return mtau, minf, htau, hinf
    
    def na_steady_soma(self,v):
        
        if v == -38:
            v += .0001

        alpha = (0.182 * (v- -38))/(1-(np.exp(-(v - -38)/6)))
        beta = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        sum = alpha + beta
        mtau = (1/sum)/self.q10
        minf = alpha/sum

        if v == -66:
            v += .0001

        alpha = (-0.015 * (v- -66))/(1-(np.exp((v- -66)/6)))
        beta = (-0.015 * (-v -66))/(1-(np.exp((-v -66)/6)))
        sum = alpha + beta
        htau = (1/sum)/self.q10
        hinf = alpha/sum

        return mtau, minf, htau, hinf
    
    def nap(self, v):
        # m ** 2

        mInf = 1.0/(1+np.exp((v- -52.6)/-4.6))
        if v == -38:
            v = v+0.0001

        mAlpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mBeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mTau = 6*(1/(mAlpha + mBeta))/self.q10

        if (v == -17) or (v == -64.4):
            v = v+0.0001
        hInf = 1.0/(1+np.exp((v- -48.8)/10))
        hAlpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17)/4.63))
        hBeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4)/2.63))
        hTau = (1/(hAlpha + hBeta))/self.q10

        return mInf, mTau, hInf, hTau


    def sk_v3_1(self,v):

        ntau = 0.2*20.000/(1+np.exp(((v -(-46.560))/(-44.140))))
        ninf = 1/(1+np.exp(((v -(18.700))/(-9.700))))

        return ntau, ninf
    
    def k_tst(self, v):
        # m ** 4
        v = v + 10
        mInf =  1/(1 + np.exp(-(v+0)/19))
        mTau =  (0.34+0.92*np.exp(-((v+71)/59)**2))/self.q10
        hInf =  1/(1 + np.exp(-(v+66)/-10))
        hTau =  (8+49*np.exp(-((v+73)/23)**2))/self.q10

        return mInf, mTau
    
    def k_pst(self, v):
        # m**2
        v = v + 10
        mInf =  (1/(1 + np.exp(-(v+1)/12)))
        if v < 50: 
            mTau =  (1.25+175.03*np.exp(-v * -0.026))/self.q10
        else:
            mTau = ((1.25+13*np.exp(-v*0.026)))/self.q10
        
        hInf =  1/(1 + np.exp(-(v+54)/-11))
        hTau =  (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/self.q10

        return mInf, mTau
    
    def Ih(self, v):
        if v == -154.9:
            v += 0.0001
        
        mAlpha =  0.001*6.43*(v+154.9)/(np.exp((v+154.9)/11.9)-1)
        mBeta  =  0.001*193*np.exp(v/33.1)
        mInf = mAlpha/(mAlpha + mBeta)
        mTau = 1/(mAlpha + mBeta)
        return mInf, mTau
    
    def Im(self, v):

        mAlpha = 3.3e-3*np.exp(2.5*0.04*(v - -35))
        mBeta = 3.3e-3*np.exp(-2.5*0.04*(v - -35))
        mInf = mAlpha/(mAlpha + mBeta)
        mTau = (1/(mAlpha + mBeta))/self.q10

        return mInf, mTau
    
    def Ca_HVA(self, v):
        
        if v == -27:
            v += 0.0001
            
        mAlpha =  (0.055*(-27-v))/(np.exp((-27-v)/3.8) - 1)        
        mBeta  =  (0.94*np.exp((-75-v)/17))
        mInf = mAlpha/(mAlpha + mBeta)
        mTau = 1/(mAlpha + mBeta)
        hAlpha =  (0.000457*np.exp((-13-v)/50))
        hBeta  =  (0.0065/(np.exp((-v-15)/28)+1))
        hInf = hAlpha/(hAlpha + hBeta)
        hTau = 1/(hAlpha + hBeta)

        return mInf, mTau, hInf, hTau
    
    def Ca_LVA(self, v):
        
        v += 10
        
        mInf = 1.0000/(1+ np.exp((v - -30.000)/-6))
        mTau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/self.q10
        hInf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
        hTau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/self.q10

        return mInf, mTau, hInf, hTau
    
    def cai_dyn(self, ica, gamma, decay):

        faraday = h.FARADAY
        mincai=1e-4
        depth = 0.1

        cai = -(10000)*(ica*gamma/(2*faraday*depth)) - (cai - mincai)/decay

        return cai
    
    def sk_e2(self, cai):

        if cai < 1e-7:
            cai += 1e-7
        
        zInf = 1/(1 + (0.00043/ cai)**4.8)
        
        return zInf
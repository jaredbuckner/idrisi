## Additional random number mechanisms

import math
import random

class JRandom(random.Random):
    def pickint(self, s):
        '''If string s is formatted "<int>:<int>", pick an integer between the bounds,
        inclusively.  Otherwise, convert s to an integer and return.
        '''
        pieces = tuple(s.split(':', 2))
        if(len(pieces) == 2):
            return(self.randint(int(pieces[0]), int(pieces[1])))
        else:
            return(int(s))
        
    def pickfloat(self, s):
        '''If string s is formatted "<float>:<float>", pick a float between the bounds,
        inclusively.  Otherwise, convert s to a float and return.
        '''
        pieces = tuple(s.split(':', 2))
        if(len(pieces) == 2):
            return(self.uniform(float(pieces[0]), float(pieces[1])))
        else:
            return(float(s))

    def rand_sum_to_n(self, summa, elements):
        '''Yield a sequence of floats "elements" long which sum to "summa"
        '''
        for pick in range(elements-1, -1, -1):
            if pick==0:
                yield summa
            else:
                x = summa - self.uniform(0,summa**pick)**(1/pick)
                summa -= x
                yield x

    def punctillate_rect(self, pMin, pMax, distsq):
        '''Yield random points in the rectangle bounded from pMin to pMax, such that
        the final number of points are on average math::sqrt(distsq) apart.
        
        pMin : (xMin, yMin)
        pMax : (xMax, yMax)
        '''
        xdist = pMax[0] - pMin[0]
        ydist = pMax[1] - pMin[1]

        for idx in range(int(xdist * ydist / distsq) + 3):
            yield (self.uniform(pMin[0], pMax[0]),
                   self.uniform(pMin[1], pMax[1]))

    def tonal_rand(self, rMin, rMax, overtones, *,
                   tiltangle=None, tiltspread=math.pi/3, istilttoward=True):
        '''Create (base, evenAmplSeq, oddAmplSeq) based on:
          
          rMin :       the minimum radius
          rMax :       the maximum radius
          overtones :  the number of entries in the *AmplSeq
          tiltangle :  interference is generated toward this tilt angle (in
                       radians).  Defaults to None, which is randomly spread
          tiltspread : the amount the interference is spread around the tilt
                       angle, if non-None.  Defaults to pi/3, corresponding to
                       one sixth of the circle.
          istilttoward : If true, interference is constructive.  If false,
                       interference is destructive.
        
        This doesn't quite honor rMin/rMax, because finding zeroes of
        high-order polynomials is hard.  We just kinda sorta try.
        '''
        base = (rMax + rMin) / 2.0;
        ampl = (rMax - rMin) / 2.0;
        thetamin = tiltangle - tiltspread / 2 if tiltangle is not None else -math.pi
        thetamax = tiltangle + tiltspread / 2 if tiltangle is not None else +math.pi
        evens = []
        odds = []
        for amplsq in self.rand_sum_to_n(ampl * ampl, overtones):
            localampl = math.sqrt(amplsq)
            theta = self.uniform(thetamin, thetamax)
            evens.append(localampl * math.cos(theta))
            odds.append(localampl * math.sin(theta))

        return(base, evens, odds)

    def koch_path(self, pFrom, pTo, maxStepSq, bendFactor):
        '''Generate points from pFrom to pTo, no farther apart than
        math.sqrt(maxStepSq), for which each sequence of three points bends no
        sharper than bendFactor, where a bendFactor of 1 is a 60-degree angle.
        
        In accordance with python range behavior, the final point (pTo) is not
        yielded.
        '''        
        pDel = (pTo[0]-pFrom[0], pTo[1]-pFrom[1])
        if pDel[0]*pDel[0] + pDel[1]*pDel[1] <= maxStepSq:
            yield pFrom
        else:
            localFactor = self.uniform(-bendFactor, bendFactor)
            pOrtho = (-pDel[1] * localFactor * math.sqrt(3.0) / 2.0,
                      pDel[0] * localFactor * math.sqrt(3.0) / 2.0)
            pMid = ((pFrom[0] + pTo[0]) / 2.0 + pOrtho[0],
                    (pFrom[1] + pTo[1]) / 2.0 + pOrtho[1])
            yield from self.koch_path(pFrom, pMid, maxStepSq, bendFactor)
            yield from self.koch_path(pMid, pTo, maxStepSq, bendFactor)

    
    def koch2_path(self, pFrom, pTo, maxStepSq, *, fixedR=None, leanLeft=None, leanRecurse=False):
        '''Generate points from pFrom to pTo, no farther apart than
        math.sqrt(maxStepSq) If a fixedR is given it should be betweeon 1/5 (a
        straight line) and 1/3 (a step function)
        
        If leanLeft is None, each shape leans in a random behavor.  If True,
        the first fractalization leans leftward. If False, the first
        fractalization leans rightward.  The leanRecurse keyword controls
        whether recursive paths also lean, or are randomized.
        
        In accordance with python range behavior, the final point (pTo) is not
        yielded.

        '''
        pDel = (pTo[0] - pFrom[0], pTo[1] - pFrom[1])
        if pDel[0]*pDel[0] + pDel[1]*pDel[1] <= maxStepSq:
            yield pFrom
        else:            
            r = self.uniform(1/5, 1/3) if fixedR is None else fixedR
            x = 1/2 - 3/2 * r
            ysq = max(0, -r*r*5/4 + r*3/2 - 1/4)
            y = math.sqrt(ysq)
            lean = self.choice((True, False)) if leanLeft is None else leanLeft
            nextLeanLeft = None if leanLeft is None or leanRecurse is False else leanLeft
            nextLeanRight = None if leanLeft is None or leanRecurse is False else not leanLeft
            pOrtho = (y * pDel[1], y * -pDel[0]) if lean else (y * -pDel[1], y * pDel[0])
            
            pAlfa = (pFrom[0] + r * pDel[0],
                     pFrom[1] + r * pDel[1])
            pBrav = (pFrom[0] + (r+x) * pDel[0] + pOrtho[0],
                     pFrom[1] + (r+x) * pDel[1] + pOrtho[1])
            pChar = (pFrom[0] + (r+r+x) * pDel[0] + pOrtho[0],
                     pFrom[1] + (r+r+x) * pDel[1] + pOrtho[1])
            pDelt = (pFrom[0] + (r+r+x+x) * pDel[0],
                     pFrom[1] + (r+r+x+x) * pDel[1])
            yield from self.koch2_path(pFrom, pAlfa, maxStepSq, fixedR=fixedR, leanLeft=nextLeanLeft, leanRecurse=leanRecurse)
            yield from self.koch2_path(pAlfa, pBrav, maxStepSq, fixedR=fixedR, leanLeft=nextLeanRight, leanRecurse=leanRecurse)
            yield from self.koch2_path(pBrav, pChar, maxStepSq, fixedR=fixedR, leanLeft=nextLeanLeft, leanRecurse=leanRecurse)
            yield from self.koch2_path(pChar, pDelt, maxStepSq, fixedR=fixedR, leanLeft=nextLeanRight, leanRecurse=leanRecurse)
            yield from self.koch2_path(pDelt, pTo, maxStepSq, fixedR=fixedR, leanLeft=nextLeanLeft, leanRecurse=leanRecurse)
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:44:27 2018
@author: dkrajnov
Potsdam 04.01.2021


#######################################################################

 Copyright (C) 2005-2021, Davor Krajnovic

 If you have found this software useful for your
 research, I would appreciate an acknowledgment to use of the
 `Kinemetry: a generalisation of photometry to the higher moments
 of the line-of-sight velocity distribution' by Krajnovic et al. (2006)'

 This software is provided as is without any warranty whatsoever.
 Permission to use, for non-commercial purposes is granted.
 Permission to modify for personal or internal use is granted,
 provided this copyright and disclaimer are included unchanged
 at the beginning of the file. All other rights are reserved.
 
 The software was originally written in IDL, here is a python
 translation (with a few modification).

#######################################################################

NAME:
   KINEMETRY

PURPOSE:
   Perform harmonic expansion of 2D maps of observed kinematic
   moments (velocity, velocity dispersion, h3, h4...) along the best
   fitting ellipses (either fixed or free to change along the radii).

EXPLANATION:
   This program is a generalisation of the ellipse fitting method used
   in photometry (e.g. Jedrzejewsky 1987) to the odd moments of the 
   line-of-sight velocity distribution (LOSVD). The even moments are 
   treated as in photometry, while there is a modification for the 
   odd moments. The method assumes that along the best fitting ellipse
   the odd moment is well approximated by a simple cosine term, as in 
   the tilted ring method by Begeman (1987). We use interpolation to 
   sample the kinematic moment along the ellipses, as done in all 
   ellipse-fitting implementations for photometry, in the few-pixels
   regime.

   For a given radius a best fit ellipse is described by flattening
   'q' (= 1 - ellipticity) and its position angle 'PA'. The ellipse
   parameters are found similarly to Jedrzejewsky (1987) photometry
   approach, but in the case of odd moments by minimising 'a1', 'a3',
   and 'b3' coefficients (or coeffs. next to sin(x), sin(3x), cos(3x))
   of the Fourier expansion of a kinematic profile extracted along an 
   ellipse. This is possible because a small error in q produces a 
   non-zero b3 coefficient, and a small error in PA produces non-zero 
   a1, a3, and b3 coefficients. Errors in the position of the centre 
   produces nonzero even terms (a2,b2), which are used if the fitting
   of the centre is also required.

   The determination of q and PA is done in two steps. First a
   grid of q and PA is specified (not very dense, but covering the
   whole parameters space) and for each combination of the
   ellipse parameters a kinematic profile is extracted. Using a
   least-squares fit (singular value decomposition) the Fourier
   coefficients are determined. The combination (q,PA), for which
   a1^2+a3^2+b3^2 is the smallest, is used as the input for the
   second step, which consists in a non-linear search for best
   parameters performed by MPFIT.

   After determination of the best fitting ellipse parameters
   (described by q and PA for a given radius), a new Fourier expansion
   is performed. The results are the Fourier coefficients and 
   reconstructed kinematic moment maps.

CALLING SEQUENCE:
    k = kinemetry(xbin=xbin, ybin=ybin, moment=moment, img=None, x0=0., y0=0.,
             ntrm=6, error=None, scale=0.8, nrad=100, name=None, paq=None, 
             npa=21, nq=21, rangeQ=None, rangePA=None, allterms=False, 
             even=False, bmodel=False, ring=None, radius=None, cover=0.75, 
             plot=True, verbose=True, nogrid=False, fixcen=True, badpix=None,
             sky=None, vsys=None)
    
    See usage example in run_kinemetry_examples.py

INPUTS:
   XBIN   - 1D array with X coordinates describing the map
   YBIN   - 1D array with Y coordinates describing the map
   MOMENT - 1D array with kin.moment (e.g. velocity) values 
            at XBIN, YBIN positions

OPTIONAL INPUTS KEYWORDS:   
   NTRM  - scalar specifying the number of terms for the harmonic analysis 
           the profile extracted along the best fitting ellipse. Default 
           value is 6 *odd* terms, which means the following terms will be 
           used in the expansion: a1, b1, a3, b3, a5 and a5, or the terms
           corresponding to sin(x),cos(x),sin(3x),cos(3x),sin(5x),cos(5x).
           To fit first 10 odd and even terms (e.g. sin(x),cos(x),
           sin(2x),cos(2x),sin(3x),cos(3x),sin(4x),cos(4x),sin(5x),cos(5x))
           set nterm=10 and even=True.
   ERROR - 1D array with errors to VELBIN values. If this keyword is 
           specified then the program will calculate formal (1sigma) errors 
           of the coefficients which are returned in ER_PA, ER_Q (determined 
           by MPFIT), ER_CF (determined by SVD) variables. IF IMG keyword is 
           set, ERROR has to be a 2D array. If if is not supplied, it is 
           creatred as 2D array with all values equal to 1. 
   SCALE - scalar specifying the pixel scale on the map (e.g. 0.8 arcsec/pixel
           for SAURON, 0.2 arcsec/pixel for MUSE). Default is 1. It has to be 
           set to 1, when passing externally determined radii (and optionally 
           PA and Q). 
   IMG   - 2D array containing an image. This keyword was designed specifically
           for surface photometry (analysis of the zeroth moment of the LOSVD), 
           but kinematic map can be also passed through this keyword. To use 
           kinemetry for photometry it is also necessary to set keyword EVEN 
           and it can be useful to use NTRM=10 in order to measure disky/boxy 
           deviations (4th terms). 
           IF IMG is set, ERROR should also be a 2D array of the same size as 
           IMG. When IMG is used, VELCIRC and VELKIN keywords contain 
           reconstructed images. It is assumed that image coordiantes are in 
           pixels (not physical units). Keyword SCALE is set to 1.  
           Internally, 2D image is reshaped into 1D arrays in this way:
                   ny,nx=img.shape
                   x = (np.arange(0,nx))
                   y = (np.arange(0,ny))
                   xx, yy = np.meshgrid(x, y)
                   xbin = xx.ravel()
                   ybin = yy.ravel()
                   moment = img.ravel()
           and the analysis is equivalent to the standard way. 
    X0   - an estimate of the X coordinate of the center, in same units as
           XBIN and YBIN. IF using IMG, X0 is in pixles. If not given X0=0. 
           For accurate determination of the center and other 
           ellipse parameters at small radii it is important the ellipse 
           includes the center of the galaxy.
   Y0    - an estimate of the Y coordinate of the center. If not given Y0=0.
   FIXCEN- set by default. When set, center will be fixed and no attempt will 
           be made during the ellipse parameter determiantion to find a new 
           center. Center is fixed to X0 and Y0. Set to False, if center 
           fitting is required. Note, finding of the centre is degenerate, 
           especially in the odd (velocity maps). For details see Krajnovic 
           et al. (2006). In order to decrease the possible degeneracies, a
           good initial estimate should be provided. 
   NRAD  - scalar specifying the number of radii along which kinemetry should 
           be run. IF not specified, NRAD=100. Kinemetry will stop when the 
           edge of the map encountered and NRAD is not necessary achived. 
           To force kinemetry to do all radii, relax condition in keyword COVER.                      
   NAME  - name of the object (used by VERBOSE keyword and for internal
           plotting)
   PAQ   - 2 element or 2*NRAD element vector specifying position angle (PA)
           and flattening (q) of the ellipses in that order (kept constant). It
           is possible to specify a set of PA and q values (that correspond to 
           given radii (see RADIUS keyword)), for which one wants to get the 
           Fourier coefficients. In this case PAQ should be set as follows: 
           PAQ=[PA1,Q1, PA2,Q2...., PAnrad,Qnrad]. It can be also used as an 
           initial condition for determination of ellipses. In this case, it 
           should be called together with NOGRID keyword (currently 
           implemented only for photometry). IF PAQ keyword is used to define 
           ellipses along which harmonic decomposition is made, then keyword 
           NOGRID should not be used. In this case center is fixed (and should 
           be defined via X0 and Y0 keywords if IMG keyword is used).
   NOGRID- if set, the direct minimisation via a grid in PA and Q values is 
           skipped. It should be used together with PAQ parameters, when
           a good estimate of PA and Q are passed to the program, but not if 
           PAQ serve to pre-define ellipses for harmonic decomposition.
           It is desigend with photometry in mind, where the problem usually 
           has only one well defined minimum (in PA,Q plane). It speeds up the
           calculation, but for the higher kinematic moments it is not robust 
           and not advised.
   NPA   - scalar specifying the number of PA used to crudely estimate
           the parameters of the best fit ellipse before entering MPFIT. 
           Default value is 21. To speed up the process and for quick tests 
           it is useful to use a small number (e.g 5). Complicated maps may 
           require a bigger number (e.g. 41).
   NQ    - scalar specifying the number of q used to crudely estimate the 
           parameters of the best fit ellipse before entering MPFIT. See NPA. 
   RANGEQ- 2 element vector specifying the min and max value for flattening Q. 
           Default values are 0.2 and 1.0 for kinematic maps and 0.1-1 for
           photometry.
   RANGEPA-2 element vector specifying the min and max value for position 
           angle PA. Default values are -90 and 90 (degrees).
;   BADPIX- 1D array containing indices of pixels which should not be used 
;           during harmonic fits. This keyword is used when data are passed via
;           IMG. It is usefull for masking stars and bad pixels. When used, 
;           XBIN and YBIN should be real coordinates of the pixels of IMG 
;           (see IMG for more details). The bad pixels are passed to the 
;           routine which defines/selects the ellipse coordiantes (and values) 
;           to be fitted, and they are removed from the subsequent fits. All 
;           pixels of the ellipse which are 2*da from the bad pixels are removed 
;           from the array, where da is the width of the ring. 
  ALLTREMS-if set then the harmonic analysis of the rings will include both 
           even and odd terms. If this keyword is set, and NTRM = n then the 
           following terms are used in the expansion: a1, b2, a2, b2, a3, 
           b3,...., an, bn (or coeffs nex to: sin(x),cos(x),sin(2x),cos(2x),
           sin(3x),cos(3x),...,sin(nx),cos(nx))
   EVEN  - set this keyword to do kinemetry on even kinematic moments. In this 
           case, kinemetry reduces to photometry and the best fitting ellipse 
           is obtained by minimising a1, b1, a2, b2 terms. When this keyword 
           is set, keyword /ALL is automatically set and NTRM should be 
           increased (e.g. NTRM=10 will use the following terms in the 
           expansion: a1, b2, a2, b2, a3, b3, a4, b4 (or coeffs. next to 
           sin(x),cos(x),sin(2x), cos(2x),sin(3x),cos(3x),sin(4x),cos(4x)))
   VSYS  - if set, the zeroth term (a0) is not extracted (applicable only for 
           odd moments - do not use with EVEN keyword). This might be useful 
           for determinatio of rotation curves. One can first run kinemetry 
           without setting this keyword to find the systemic velocity 
           (given as cf[*,0]). Then subtract the systemic velocity form the 
           velocity map and re-run kinemetry with vsys set (vsys=0). In this 
           case the zeroth terms will be excluded from the Fourier analysis. 
           For completeness, it is also possible to input VSYS, e.g. VSYS=2000. 
           The zeroth term will not be calculated, but it will be set to 2000
           in output. Given that Fourier terms are orthogonal, it should not be 
           necessary to set this keyword in general. Note that the fit to the
           velocities along the ellipse will generally be worse (not good!) 
           when this keyword is set, but the ellipse parameters (PA, Q) will
           be correct. 
   RING  - scalar specifying desired radius of the first ring. Set this keyword 
           to a value at which the extraction should begin. This useful in case
           of ring-like structures sometimes observed in HI data.
  RADIUS - 1D array with values specifying the lenght of the semi-major axis
           at which the data (kin.profile) should be extracted from the map
           for the kinemetric analisys. The values should be in pixel space 
           (not in physical units such as arcsec). If this keyword is set,
           the values are coopied into the output variable: RAD.
   COVER - Keyword controling the radius at which extraction of values from 
           the map stops. Default value is 0.75, meaning that more than 
           75% of the points along an ellipse are have to be present, otherwise
           the kinemetry is stops. The value of 75% is an hoc, but a 
           conservative value which ensures that kinemtric coeffs. are robust 
           (at least the lower few orders). Sometimes it is necessary to relax
           this condition, especially for the reconstruction of maps. Cover=1
           means all ellipse points have to be on the map, while for cover=0.2
           only 20% of the ellipse points have to be on the map.  
   BMODEL- if set (default True), 2D models of the   moment map can be constructed.
          otherwise kinemetry will return moments sampled along the best ellipse
          (see OUTPUTS). If set, VELCIRC, VELKIN and GASCIRC outputs will 
          contain the reconstructed maps, using the first dominant term and 
          all terms, respectively. IF IMG keyword is used, the outputs are 2D 
          images, otherwise BMODEL reconstructs the map at each input position 
          XBIN,YBIN.If BMODEL is False VELCIRC and VELKIN will contain 
          reconstructed values at the positions of XELLIP and YELLIP.
   PLOT - If this keyword is set, diagnostic plots are shown for each radii: 
		       - the best ellipse (overploted on kin.moment map). If IMG
		         keyword is set, the image is scaled to the size of the ellipse.
		         The centering of the overplotted ellipse is good to 0.5 pixels
		         so for small radii (r < a few pixels) it is possible that the 
		         position of the center of the overplotted ellipse is not on the
		         brightes pixel. 
		       - PA-Q grid for the cacualtion of the initial values of the ellipse
                 position angla and axial ratio used by the MPFIT minimisation.
                 Colours show linearly interpolated distribtuion of a sum in 
                 quadrature of a1,a3,and b3 (odd case) or a1,b1,a2 and b2 
                 (even case). The minium value of this sum determines the (PA,Q)
                 pair that will define the best fitting ellipse. Plot shows
                 with a black circles the location of the minimum on the grid, 
                 which is used as an input to the second level minimsation with 
                 the MPFIT. Red cirlce show the final PA,Q values which define 
                 the ellipse. 
		       - fit to kin.profile (points are the DATA, red is the kinemetry
                 fit based on all terms, blue is the kinemetry fit based on only 
                 a0+b1*cos(x) for odd, and a0 for even moments. Light blue is 
                 the same fit, but extrapolated to where there are no data points. 
		       - residuals (DATA - FIT), and overplotted higher order
		         terms (green: a1,a3 and b3, red: a1,a3,b3,a5 and b5; 
		         for the /EVEN case - green: a1,b1,a2,b2, red:a1,b1,a2,b2,a4,b4)
 VERBOSE- set this keyword to print status of the fit on screen 
          including information on:
               - Radius - number of the ring that was analysed
               - RAD    - radius of the analysed ring (if SCALE is passed 
                          it is given in the same units, otherwise in pixels)
               - PA     - position angle of the best fitting ellipse
               - Q      - flattening of the best fitting ellipse
               - Xcen   - X coordinate of the centre of the ellipse
               - Ycen   - Y coordinate of the centre of the ellipse
               - # of ellipse elements - number of points to which the data
                          points in the ring are sampled before derivation of
                          the best fit parameters and harmonic analysis. It 
                          can vary between 20 and 100 in non IMG model
                          depending on the ring size, giving a typical sampling 
                          of 3.6 degrees.
OUTPUTS, as part of class k:
   RAD    - 1D array with radii at which kin.profiles were extracted
   PA     - 1D array with position angle of the best fitting ellipses,
            PA is first determined on an interval PA=[-90,90], where
            PA=0 along positive x-axis. Above x-axis PA > 0 and below
            x-axis Pa < 0. PA does not differentiate between receding
            and approaching sides of (velocity) maps. This is
            transformed to the usual East of North system, where the 
            East is the negative x-axis, and the North is the positive 
            y-axis. For odd kin.moments PA is measured from the North 
            to the receding (positive) side of the galaxy (which is 
            detected by checking the sign of the cos(theta) term. For
            the even terms it is left degenerate to 180 degrees rotation.
   Q      - 1D array with flattening of the best fitting ellipses
            (q=1-ellipticity), defined on interval q=[0.2,1]
   CF     - 2D array containing coefficients of the Fourier expansion
            for each radii cf=[Nradii, Ncoeff]. For example: 
	           a0=cf[:,0], a1=cf[:,1], b1=cf[:,2]....
    XC    - the X coordinate of the center (in same units as XBIN). If X0 not fit XC=0.
    YC    - the Y coordinate of the center (in same ubuts as XBIN). If Y0 not fit YC=0.
   ER_PA  - 1D array of 1 sigma errors to the ellipse position angle
   ER_Q   - 1D array of 1 sigma errors to the ellipse axial ratio
   ER_CF  - 2D array containing 1 sigma errors to the coefficients 
            of the Fourier expansion for each radii
   ER_XC  - the X coordinate of the center (in pixels). If X0 not fit XC=0.
   ER_YC  - the Y coordinate of the center (in pixels). If Y0 not fit YC=0.

   VELCIRC -1D array containinng 'circular velocity' or a0 + b1*cos(theta)
            at positions XBIN, YBIN (velcirc = a0, in case of EVEN moments), 
	        obtained by linear interpolation from points given in XELLIP 
            and YELLIP keywords, if BMODEL keyword is set (default)), otherwise
            at positions XELLIP and YELLIP.
   VELKIN - 1D array of reconstructed kin.moment using NTRM harmonic
            terms at positions XBIN,YBIN, obtained by linear interpolation
            from points given in XELLIP and YELLIP keywords, if BMODEL keyword
            is set (default), otherwise at positions XELLIP and YELLIP.
   GASCIRC -1D array containing circular velocity or Vcirc=cf[*,2]*cos(theta)
            at positions XBIN and YBIN, obtained for XellipC,YellipC based
            on fixed PA and q. PA and q are taken to be median values of the 
            radial variation of PA and q. IF keyword PAQ is used than GASCIRC 
            give the circular velocity (no systemic velocity) for the median 
            values of PA, Q values. Note that this is different
            from VELCIRC (also VELKIN) which is obtained on the best
            fitting ellipses and also includes Vsys (cf[*,0]) term. 
            This keyowrd is useful for gas velocity maps, if one wants to obtain 
            a quick disk model based on the circular velocity. 
   VSYS  -  if set in the call to kinemetry, it returns the input value, 
            otherwise is None.
   XELLIP - 1D array with X coordintes of the best fitting ellipses
   YELLIP - 1D array with Y coordintes of the best fitting ellipses
   XELLIPC- 1D array with X coordintes of the ellipses with fixed PA and Q, used for gascirc 
   YELLIPC- 1D array with Y coordintes of the ellipses with fixed PA and Q, used for gascirc 
   ECCANO - eccentric anomaly of the best ellipses
   EX_MOM - extracted "data" along the best fit ellispes (same size as ECCANO)
   nelem  - number of eccentric anomaly points for each extracted ellipse
   vv     - circular velocity along the best fitting ellipse 
   vrec   - full kinemetry reconstruction along the best fitting ellipse 
   vvF    - "gas" circular velocity along the ellipse

RESTRICTIONS:
   Running the python version of kinemetry on the example velocty map 
   (SAURON, with 708 data points) takes 118 sec on 2.8GHz MacBook pro, 
   and it is about 10 times slower than the IDL version. The decreese 
   in speed is related to the speed of the interpoaltion routine used 
   by python to interpolate the irregular Voroni grid of the map to a 
   sequence of points along an ellipse. Speed decreases further by the 
   size of the map. 
   
   Speed and robustness of the program also depend on the number of NQ
   and NPA that define the initial PA,Q grid used to determine the location
   of the local minima. Small grid is fast, but not precise for compplex maps. 
   In the case of IMG keyword, plotting the results on the screen is a also 
   a significant contributor to the decrease in speed.

REQUIRED ROUTINES:
   The following routines are need and available on Michele Cappellari's
   Python Pacakage Index pages: https://pypi.org/user/micappe/
   - plot_velfield within plotbin
   - cap_mpfit within mgefit (mpfit was converted to IDL C.B. Markwardt 
     http://astrog.physics.wisc.edu/~craigm/idl/ and subsequently translated 
     to python by Mark Rivers, with further modifications by Sergey Koposov)

EXAMPLES:
    See also example script: run_kinemetry_examples.py
    
    1) 
    Run kinemetry on a velocity map with pixel scale of 0.2", defined with 
    coordinate arrays: Xbin, Ybin, and a velocity array: Velbin. 
 
    k = kinemetry(Xbin, Ybin, Velbin, scale=0.2)
    
    Desired outputs are stored in k as a structure: position angle and 
    flattening of the ellipses, harmonic terms: a0, a1, b1, a3, b3, a5, b5 
    and a reconstructed map of the circular velocity. They are storred as: 
        k.pa, k.q, k.cf[:,0], k.cf[:,1], k.cf[:,2], k.cf[:,3], k.cf[:,4]
        k.cf[:,5], k.cf[:,6], k.velcirc
        
    2)
    Run kinemetry on a velocity map starting at radius=5". Desired outputs 
    are a set of radii with position angles and flattenings of the ellipses, 
    harmonic terms: a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a 
    reconstructed map of the circular velocity and a map reconstructed with 
    all terms: 

    k = kinemetry(Xbin, Ybin, Velbin, nterm=10, allterms=True, ring=5)

    Outputs are: k,rad, k.pa, k.q, a0=k.cf[:,0], a1=k.cf[:,1], b1=k.cf[:,2], 
        a2=k.cf[:,3], b2=k.cf[:,4], a3=k.cf[:,5], b3= k.cf[:,6],
        a4=k.cf[:,7], b4=k.cf[:,8], a5=k.cf[:,9], b5=k.cf[:,10], k.velcirc
        k.velkin

    3) 
    Run kinemetry on a velocity dispersion map (given by array Sigma). 
    Desired outputs are position angle and flattening of the ellipses,
    and harmonic terms: a0, a1, b1, a2, b2, a3, b3, a4, b4:

    k = kinemetry(Xbin, Ybin, Sigma, ntem=10, even=True)

    4)
    Run kinemetry on an image of a galaxy to perform surface photometry.
    Image is given with 2D array IMG. X0 and Y0 are estimates of the center
    of the galaxy. X,Y,FLUX are dummy 1D arrays. They are still required, but are
    irrelevant if IMG keyword is used and can be anything. 

    k = kinemetry(X, Y, FLUX, img=IMG, x0=X0, y0=Y0, nterm=10, EVEN)


    5) 
    Run kinemetry on an image of a galaxy to perfrom surface photometry. An 
    image is given with 2D array IMG. X0 and Y0 are estiamtes of the center
    of the galaxy. X,F,FLUX are 1D dummy arrays. Determination of PA and Q
    via a grid is skipped (nogrid=True), but an initial estimate of PA and Q
    must be passed bia PAQ keyword. Outputs are for each ellipse of semi-major 
    lenght RAD (XELLIP, YELLIP): center coordinates (XC,YC), PA, Q, CF and
    associated errors. Models are made reconstructed on each image pixel. 
    
       k = kinemetry(X, Y, flux, img=IMG, nterm=10, even, 
                     x0=X0, y0=Y0, nogrid=True, PAQ=[ang-90, 1-epsI])
    
REVISION HISTORY:
   V1.0 - Written by Davor Krajnovic and Michele Cappellari (March, 2005)
   V2.0 - First released version, Davor Krajnovic, Oxford, 7.12.2005. 
   V2.1 - Add keyword RADIUS, DK, Oxford, 27.01.2006. 
   v2.2 - Changed definition of PAQ keyword, DK, Oxford, 02.02.2006.
   v2.3 - Corrected bug: keyword \even was not pass correctly to MPFIT. 
	         Thanks to Roland Jesseit for prompting. Minor homogonisation 
          of the code. DK, Oxford, 06.02.2006
   V2.4 - Add keyword RANGEPA, DK, Oxfrod, 11.07.2006.
   V2.5 - Add keyword COVER, DK, Zagreb, 02.11.2006.
   v2.6 - Introduced standard definition of coeffs. for odd kinematic
	         moments: they are measured from receding side of galaxy.
	         Thanks to Anne-Marie Weijmans for asking. DK, La Palma, 24.04.2007.
   v3.0 - Adaptation of kinemetry to efficiently work on images of galaxies
          for doing surface photometry. New keywords included are IMG, X0,Y0,
	         NRAD, BMODEL, XC,YC, ER_XC, ER_YC. kinem_extract_sector.pro was added, 
	         while the main procedure and kinem_fitfunc_ellipse.pro were modified.
	         Present distribution is fully compatible with the old one, with one 
	         exception: VELCIRC and VELKIN are now returend as reconstructions at
	         XELLIP and YELLIP positions unless keyword BMODEL is set as well. In 
	         that case VELCIRC and VELKIN are reconstruceted as map. For the moment
	         XBIN, YBIN and MOMENT still need to be passed at the same time as IMG, 
	         if IMG is set they are not used at all and can be anything. This will 
	         be change in future distributions. DK, Oxford, Queens College, 16.11.2007.
   v3.1 - Clearing of a bug for using IMG for ODD moments. 2D images of both even (flux) 
          and odd moments (e.g. velocity) can be analysied. Center is STILL not fitted 
          for ODD moments. Thanks to Maxim Bois for prompting. DK, Hatfield, 03.09.2008.
   v3.2 - Passing center values when no minimization is attempted and PAQ values are 
          kept constant. DK, Oxford, 31.10.2008.
   v3.3 - Introduced FIXCEN for EVEN moments. Sorted bug related to RADIUS and IMG
          keywords. Thanks to Kristen Shapiro. DK, Hatfield, 03.11.2008.
   v3.4 - Sort some minor bugs related to FIXCEN. Thanks to Kristen Shapiro. 
          DK, Hatfield, 17.12.2008.
   v3.5 - Introduced keyword BADPIX, Oxford, DK, 20.02.2009.
   v3.6 - Introduced keyword GASCIRC and VSYS. Thanks to Witold Maciejewski. 
          DK, Muechen, 08.12.2009.
   v3.7 - Major testing. In IMG model ellipse sampling was changed to depend on 
          the radius, but with the maximum of 64 points, affecting first 5 rings. 
          For nonIMG mode the maximum wasleft to 100 as before. Cleaning of the
          documentation. DK, Muenchen, 13.05.2010. 
   v3.8 - Adjusted plotting of the map in /even case (but no IMG) for diagnostic
          plots (when /plot set). DK, Garching, 20.05.2010.
   v3.9 - Added SKY keyword to enable stopping at the sky level. DK, Munich, 26.10.2010.
   v4.0 - Introduced GRIDDATA instead of a call to TRIGRID in the function kinem_trigrid_irregular, 
          following a suggestion from Michele Cappellari. This speeded up the the evaulation
          of the gird significantly (~6 times for a SAURON map). 
        - Fixed fitting with a free centre for the EVEN case. 
        - Re-structured calls to the initiatlization of parameters for MPFIT (parinfo)
          for various input keywords. 
          DK, Potsdam, 10.10.2013.
   v4.1 - Removed a conflict on defining parfino for the call to MPFIT, in case of 
          photometry and even kinematics moments (sigma)
          DK, Potsdam 10.03.2015.
   v4.2 - Removed a bug in assigning errors to xc and yc in the photometry mode (when using IMG). 
          Thanks to Juan Carlos Basto Pineda. 
          DK, Potsdam 04.04.2016.
   v4.3 - Changed the automatic stopping of the fit in the case of IMG keyword (isophotometry). 
          The program will now stop when the semi-major axis is 10% larger than larger side of
          the image. 
          DK, Potsdam, 08.12.2016.
   v4.4 - Added keywords: ex_mom, eccano and nelem. They can be used to extract data
          along each best fit ellipse.
   v5.0 - Conversion to python, clean up and overahual of all routines. There are some
          small difference between IDL and python versions for calling kinemetry. Check
          the script withe examples to see the simplest way to run kinemetry. 
          DK, Potsdam 01.03.2021.
   v5.1 - Made compatible with cap_mpfit.py v1.2.3 (17 Jan 2020) of Michele Cappellari, 
          which is distrubuted with mgefit package (https://pypi.org/project/mgefit/).
          This version of cap_mpfit will be distrubuted with kinemetry package to ensure 
          compatibility. It is renamed to kin_mpfit.py to avoid confusion with Michele's 
          version.
          
          

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from plotbin.plot_velfield import plot_velfield
from matplotlib.patches import Ellipse

from kin_mpfit import mpfit
import matplotlib.ticker
import matplotlib as mpl

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


# used for debugging
# import pdb #use with pdb.set_trace()
# import time

# ----------------------------------------------------------------------------
def kin_range(x1, x2, n):
    # retunrs an array of n numbers between x1 and x2 values

    val = x1 + (x2 - x1) / (n - 1) * np.arange(n)

    return val


# ----------------------------------------------------------------------------
def kinem_trigrid_irregular(xbin, ybin, moment, xnew, ynew, missing=None):
    """
     Given a set of irregular input coordinates and corresponding values for
     output coordinates, which are also irregular, it gives an interpolated 
     values at the output positions. Used for interpolating back to the original
     grid of data points (IFU maps)
    """

    assert xbin.size == ybin.size == moment.size, 'XBIN, YBIN and VELBIN must have the same size'

    assert xnew.size == ynew.size, 'XNEW and YNEW must have the same size'
    points = np.transpose(np.array([xbin, ybin]))
    momNew = griddata(points, moment, (xnew, ynew), method='linear', fill_value=missing)

    return momNew


# ----------------------------------------------------------------------------
def kinem_lstsq_solve(a, b, errors=False):
    """
    v1.0 Davor Krajnovic, Potsdam, 27 June 2018

    Solve a linear system using numpy linalg.lstsq
    method.  IF error keyword set, a and b arrays are 
    considered to be properly divided by 
    Gaussian errors. Uncertainties of the returned
    parameters are calculated by computing the 
    covariance matrix and taking the SQRT of 
    the diagonal elements (following svdvar routine
    from Numerical Recepies, Press et al. 1992). 

    OUTPUT:
        weights - coefficients of the linear system
        perr - errors on the coefficients

    """

    sa = a.shape
    sb = b.shape
    assert sa[0] == sb[0], 'LSSQT Solve Error: incompatible dimensions'

    weights = np.linalg.lstsq(a, b, rcond=None)[0]

    if errors:
        U, s, Vh = np.linalg.svd(a, full_matrices=False)
        w = s > np.finfo(float).eps * max(a.shape) * s[0]
        cov = (Vh[w].T / s[w] ** 2) @ Vh[w]
        perr = np.sqrt(np.diag(cov))
        return weights, perr
    else:
        return weights


# ----------------------------------------------------------------------------
def kinem_fit_trig_series(x, y, w, nterms, err, All=False, even=False, vsys=False):
    """
    v1.0 Davor Krajnovic, Potsdam, 27 June 2018.
         Translated from the IDL version. 
         There is one notable difference with respect to the original IDL verison:
         - the values are reconstructed for each eccentricity, even for those 
         that did not have a vaild data point as they were outside the velocity maps
         (NaNs after extraction by griddata).
                 
    
    Preparation of the set of equations of the ax=b form to be
    solved. The "coefficient" matrix a is prepared depending on the
    number of harmonic terms, and the parity of the required extraction
    (odd, even). The actual solution is provided by kinem_svd_solve function.
    If errors are passed, the errors on the harmonic coefficients will be 
    provided. 
    
    
    The harmonic series below can include 
    1) odd J values:
       1, [sin(x), cos(x)], [sin(3x), cos(3x)],...
       The even trigonometric terms are zero in the point-symmetric case.
       NTERMS=4 for all odd terms up to [sin(3x), cos(3x)] and
       NTERMS=6 for all odd terms up to [sin(5x), cos(5x)]
    2) even J values
       1, [sin(x), cos(x)], [sin(2x), cos(2x)],...
       This is used for a complete harmonic analysis, or to extract
       EVEN terms. 
       NTERMS=10 for all terms up to [sin(5x), cos(5x)]
    3) odd or even, but without zero-th term (or systemic velocity)

    OUTPUTS:
        coeff - harmonic coefficients
        yfit  - reconstrcuted values for each eccentricty which has an associated 
                data point
        yfit1 - reconstrcuted values for each eccentricty
        er_coeff - coefficient errors
    
    """

    arr = np.zeros(shape=(x[w].size, nterms + 1))
    arr1 = np.zeros(shape=(x.size, nterms + 1))  # for extrapolated values
    #
    # if keyword vsys set, do not extract zero-th term
    #
    if vsys:
        arr[:, 0] = 0.
        arr1[:, 0] = 0.
    else:
        arr[:, 0] = 1.
        arr1[:, 0] = 1.

    if (All == True) | (even == True):
        for j in np.arange(1, nterms, 2):
            arr[:, j] = np.sin(((j + 1) / 2) * x[w])
            arr[:, j + 1] = np.cos(((j + 1) / 2) * x[w])
            arr1[:, j] = np.sin(((j + 1) / 2) * x)
            arr1[:, j + 1] = np.cos(((j + 1) / 2) * x)

    else:
        for j in np.arange(1, nterms, 2):
            arr[:, j] = np.sin(j * x[w])
            arr[:, j + 1] = np.cos(j * x[w])
            arr1[:, j] = np.sin(j * x)
            arr1[:, j + 1] = np.cos(j * x)

    #
    # divide by errors
    #
    if err is None:
        coeff = kinem_lstsq_solve(arr, y, errors=False)
    else:
        y1 = y / err
        brr = arr * 0.
        for j in range(nterms + 1):
            brr[:, j] = arr[:, j] / err

        coeff, er_coeff = kinem_lstsq_solve(brr, y1, errors=True)

    yfit = np.matmul(arr, coeff)
    yfit1 = np.matmul(arr1, coeff)

    if err is None:
        return coeff, yfit, yfit1
    else:
        return coeff, yfit, yfit1, er_coeff


# ----------------------------------------------------------------------------
def kinem_fitfunc_ellipse(p, nterms, r, xbar=None, ybar=None, moment=None, interp=None,
                          even=False, allterms=False, img=None, x0=None, y0=None, badpix=None,
                          er_interp=None, vsys=None, mpf=False, grid=False, fixcen=True, fjac=None):
    """
    The main subroutine used for calling subroutines that extract values along ellipses and
    perform the harmonic expansion. 

    There are two types of outputs, depending on the use of the subrutine.
    If used to find the best fit parameters of the ellipse, the output is 
    a set of Harmonic coefficients: a1,a3,b3 in ODD case, and a1,b1,a2,b2
    in EVEN case. If using the "grid" approach, the output is a quadratic sum, 
    if using MPFIT, it is only the coefficients.
 
    If used for the harmonic analysis, outputs are:                  
        coeff - 1D array with Harmonic coefficients
        xell - 1D array with x values of points along the ellipse
        yell - 1D array with x values of points along the ellipse
        theta - 1D array with eccentricities
        momEll -1D array of extracted data values (e.g. velocities) along the ellipse
        er_momEll -1D array of extracted errors (e.g. for velocities) along the ellipse
        momFit - 1D array with reconstructed values (e.g. velocities) along the ellipse
        momFitExtr - 1D array with reconstructed values (e.g. velocities) along the ellipse,
                     also for eccentricities that are outside of the area covered by data.

    """
    ang = np.radians(p[0] - 90)
    sp = p.size

    # construction of elliptical coordinates on which kin.moment is
    # interpolated; expansion of kin.profile in harmonic series;
    # used by both 'brute force' and MPFIT minimisation

    mi = (360. / (180 / 10))
    theta = kin_range(0.0, 2 * np.pi, np.clip(mi, 10 * r, 100))
    # theta = -1*np.arctan(np.tan(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * p[1]
    if sp == 4:
        xell = p[2] + x * np.cos(ang) - y * np.sin(ang)
        yell = p[3] + x * np.sin(ang) + y * np.cos(ang)
    else:
        xell = x0 + x * np.cos(ang) - y * np.sin(ang)
        yell = y0 + x * np.sin(ang) + y * np.cos(ang)

    # call results of interpolation and interpolate to points along the ellipse
    momEll = interp(xell, yell)
    if er_interp is None:
        er_momEll = np.full_like(momEll, 1.)  # if no errors were passed
    else:
        er_momEll = er_interp(xell, yell)

    # remove "missing values"
    w = np.where(momEll != 12345678)
    if w[0].size == 0:
        return 1e30
    momEll = momEll[w]
    er_momEll = er_momEll[w]

    #
    # The Fourier analysis and the calculation of coefficients, with options for ALL, EVEN or Vsys
    #
    coeff, momFit, momFitExtr, er_coeff = kinem_fit_trig_series(theta, momEll, w, nterms, er_momEll, All=allterms,
                                                                even=even, vsys=vsys)

    if even == True:
        #
        # Fowllowing eq(1) of Jedrzejewski (1987), it tries to
        # minimize a1,a2,a2,b2, which indicate incorrect PA and
        # flattening of the trial ellipse, as well as centre c
        # coordiantes (terms are defined differently thab
        # in the 'odd' case)	
        #
        if mpf == True:
            return coeff[[1, 2, 3, 4]]  # , coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr
        elif grid == True:
            return np.sum(coeff[[1, 2, 3, 4]] ** 2)  # , coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr
        else:
            return coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr, w, er_coeff

    else:
        #
        # Fowllowing eq(1) of Jedrzejewski (1987), but for odd
        # kinematic moments, it tries to minimize a1,a3,b3, which 
        # indicate incorrect PA and flattening of the trial ellipse. 
        # In case of centre fitting, a1,a2,b2,a3 and b3 are used 
        # (see Fig. B3 in Krajnovic et al. 2006)
        #
        if (mpf == True) & (fixcen == True):
            return coeff[[1, 3, 4]]  # a1,a3,b3
        elif (mpf == True) & (fixcen == False):
            return coeff[[1, 3, 4, 5, 6]]  # a1,a2,b2, a3, b3
        elif grid == True:
            return np.sum(coeff[[1, 3, 4]] ** 2)  # a1,a3,b3
        else:
            return coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr, w, er_coeff


# ----------------------------------------------------------------------------
def kinemetry(xbin=None, ybin=None, moment=None, img=None, x0=0., y0=0.,
              ntrm=6, error=None, scale=1, nrad=100, name=None, paq=None,
              npa=21, nq=21, rangeQ=None, rangePA=None, allterms=False,
              even=False, bmodel=True, ring=None, radius=None, cover=0.75,
              plot=True, verbose=True, nogrid=False, fixcen=True, badpix=None,
              sky=None, vsys=None, output_file=None):

    ## Defining the Result Class ================================
    class Results:
        def __init__(self):
            self.type = "kinemetry analysis"

    ## ===========================================================

    ## Here is the result structure which will be filled in at the end
    results = Results()
    ## ===========================================================

    # make sure keywords are set correctly
    odd = True
    if even:
        odd = False

    #
    # check if using 2D image or 3x1D arrays, xbin,ybin,moment
    # if using 2D image, resample it to 3 1D arrays
    #
    if img is not None:
        ny, nx = img.shape
        mx_img = np.max(img.shape)
        x = (np.arange(0, nx))
        y = (np.arange(0, ny))
        xx, yy = np.meshgrid(x, y)
        xbin = xx.ravel()
        ybin = yy.ravel()
        moment = img.ravel()
        if error is not None:
            error = error.ravel()
        scale = 1
    else:
        # pdb.set_trace()
        assert xbin.size == ybin.size == moment.size, 'XBIN, YBIN and VELBIN must have the same size'

    # change to pixels
    xbar = xbin / scale
    ybar = ybin / scale
    x0 = x0 / scale
    y0 = y0 / scale

    # setting radii
    if radius is None:
        pix = np.arange(nrad)
        rad = pix + 1.1 ** pix  # geometric progression
    else:
        rad = radius
        nrad = radius.size

    #
    # The central pixel is left unchanged in the reconstruction.
    # Shifting of the first radius in case of a central hole.
    #
    if ring is not None:
        rad = ring / scale + rad
        xellip = np.zeros(1)  # np.array([])
        yellip = np.zeros(1)  # np.array([])
        eccano = np.zeros(1)  # np.zeros([])
        ex_mom = np.zeros(1)  # np.zeros([])
        vv = np.zeros(1)  # np.array([])
        vrec = np.zeros(1)  # np.array([])
    else:
        if img is not None:
            xellip = np.zeros(1)
            yellip = np.zeros(1)
            eccano = np.zeros(1)
            ex_mom = np.zeros(1)
            vv = np.zeros(1)
            vrec = np.zeros(1)
            xellip[0] = x0
            yellip[0] = y0
            vv[0] = img[int(round(x0)), int(round(y0))]
            vrec[0] = img[int(round(x0)), int(round(y0))]
        else:
            mini = np.min(np.sqrt(xbar ** 2 + ybar ** 2))
            ww = np.where(np.sqrt(xbar ** 2 + ybar ** 2) == mini)
            xellip = xbin[ww]
            yellip = ybin[ww]
            eccano = np.zeros(1)
            ex_mom = np.zeros(1)
            vv = moment[ww]
            vrec = moment[ww]

    #
    # Initialised vectors of results
    #
    pa = np.zeros(nrad)
    q = np.zeros(nrad)
    cf = np.zeros((nrad, ntrm + 1))
    xc = np.zeros(nrad)
    yc = np.zeros(nrad)
    er_cf = np.zeros((nrad, ntrm + 1))
    er_pa = np.zeros(nrad)
    er_q = np.zeros(nrad)
    er_xc = np.zeros(nrad)
    er_yc = np.zeros(nrad)
    nelem = np.zeros(nrad)

    #
    #  Initialize parameters for MPFIT in ODD case
    #
    if odd == True:
        parinfo = [{'step': 1.0, 'limits': [0.0, 0.0], 'limited': [1, 1]} for i in range(2)]
        if rangePA:
            parinfo[0]['limits'] = [rangePA[0], rangePA[1]]
        else:
            parinfo[0]['limits'] = [-95., 95.]  # PA limits in degrees
        if rangeQ:
            parinfo[1]['limits'] = [rangeQ[0], rangeQ[1]]
        else:
            parinfo[1]['limits'] = [0.2, 1.0]  # PQ limits
        parinfo[0]['step'] = 0.5  # Step in degrees (of the order of the expected accuracy)
        parinfo[1]['step'] = 0.01  # Q

    #
    # case for ODD and centre free
    #
    if (odd == True and fixcen == False):
        parinfo = [{'step': 1.0, 'limits': [0.0, 0.0], 'limited': [1, 1], 'fixed': 0} for i in range(4)]
        if rangePA:
            parinfo[0]['limits'] = [rangePA[0], rangePA[1]]
        else:
            parinfo[0]['limits'] = [-95., 95.]  # PA limits in degrees
        if rangeQ:
            parinfo[1]['limits'] = [rangeQ[0], rangeQ[1]]
        else:
            parinfo[1]['limits'] = [0.2, 1.0]  # PQ limits
        parinfo[0]['step'] = 0.5  # Step in degrees (of the order of the expected accuracy)
        parinfo[1]['step'] = 0.01  # Q
        parinfo[2]['step'] = 0.1
        parinfo[3]['step'] = 0.1
        parinfo[2]['limits'] = [x0 - 5, x0 + 5]
        parinfo[3]['limits'] = [y0 - 5, y0 + 5]

        #
    # case for higher even moments, e.g. velocity dispersion
    #
    if (even == True and fixcen == False):
        parinfo = [{'step': 1.0, 'limits': [0.0, 0.0], 'limited': [1, 1]} for i in range(4)]
        if rangePA:
            parinfo[0]['limits'] = [rangePA[0], rangePA[1]]
        else:
            parinfo[0]['limits'] = [-95., 95.]  # PA limits in degrees
        if rangeQ:
            parinfo[1]['limits'] = [rangeQ[0], rangeQ[1]]
        else:
            parinfo[1]['limits'] = [0.2, 1.0]  # PQ limits
        parinfo[0]['step'] = 0.5  # Step in degrees (of the order of the expected accuracy)
        parinfo[1]['step'] = 0.01  # Q
        parinfo[2]['step'] = 0.1
        parinfo[3]['step'] = 0.1
        parinfo[2]['limits'] = [x0 - 5, x0 + 5]
        parinfo[3]['limits'] = [y0 - 5, y0 + 5]

        #
    # keep centre fixed in photometry
    #
    if (even == True and fixcen == True):
        parinfo = [{'step': 1.0, 'limits': [0.0, 0.0], 'limited': [1, 1]} for i in range(2)]
        if rangePA:
            parinfo[0]['limits'] = [rangePA[0], rangePA[1]]
        else:
            parinfo[0]['limits'] = [-95., 95.]  # PA limits in degrees
        if rangeQ:
            parinfo[1]['limits'] = [rangeQ[0], rangeQ[1]]
        else:
            parinfo[1]['limits'] = [0.2, 1.0]  # PQ limits
        parinfo[0]['step'] = 0.5  # Step in degrees (of the order of the expected accuracy)
        parinfo[1]['step'] = 0.01  # Q

    #
    # fit for centre in photometry
    #
    if (even == True and fixcen == False and img is not None):
        parinfo = [{'step': 1.0, 'limits': [0.0, 0.0], 'limited': [1, 1], 'fixed': 0} for i in range(4)]
        if rangePA:
            parinfo[0]['limits'] = [rangePA[0], rangePA[1]]
        else:
            parinfo[0]['limits'] = [-95., 95.]  # PA limits in degrees
        if rangeQ:
            parinfo[1]['limits'] = [rangeQ[0], rangeQ[1]]
        else:
            parinfo[1]['limits'] = [0.2, 1.0]  # PQ limits
        parinfo[0]['step'] = 0.5  # Step in degrees (of the order of the expected accuracy)
        parinfo[1]['step'] = 0.01  # Q
        parinfo[2]['step'] = 0.1
        parinfo[3]['step'] = 0.1
        parinfo[2]['limits'] = [x0 - 100, x0 + 100]
        parinfo[3]['limits'] = [y0 - 100, y0 + 100]

        #
    # Set grids for global minimization
    #
    if rangeQ == None:
        if img is not None:
            rangeQ = [0.1, 1.0]
        else:
            rangeQ = [0.2, 1.0]
    if rangePA == None:
        rangePA = [-90.0, 90.0]

    #
    # speed up the calcultions, without grid, but with initial PA and q values
    # but put PAQ=0 
    #
    if paq is not None:
        if nogrid == True:
            pa_mpf = paq[0]
            q_mpf = paq[1]
            paq = None

    xpa = np.linspace(rangePA[0], rangePA[1], npa)
    xq = np.linspace(rangeQ[0], rangeQ[1], nq)
    xxpa, xxq = np.meshgrid(xpa, xq)
    pa_grid = xxpa.ravel()
    q_grid = xxq.ravel()
    chi2_grid = np.zeros(pa_grid.size)

    # triangulations
    # print('starting linear interpolation')
    # using LinearNDInterpolator
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
    if img is not None:
        points = np.transpose(np.array([xbar, ybar]))
        interp = LinearNDInterpolator(points, moment, fill_value=12345678)
        if error is not None:
            er_interp = LinearNDInterpolator(points, error, fill_value=12345678)
        else:
            er_interp = None
    else:
        points = np.transpose(np.array([xbar, ybar]))
        interp = LinearNDInterpolator(points, moment, fill_value=12345678)
        if error is not None:
            er_interp = LinearNDInterpolator(points, error, fill_value=12345678)
        else:
            er_interp = None
    # print('finished linear interpolation')

    if plot:
        f, ax = plt.subplots(figsize=(8, 9))

    k = 0
    #
    # loop over radii
    #
    for i in range(nrad):
        #
        # check if PA and q are set constant for the whole map (PAQ)
        #
        if paq is None:
            #
            # Perform brute-force *global* minimization of chi2 on a regular grid. 
            # This is needed as the problem can have multiple minima. The result
            # are PA and Q which will be used as initial values for MPFIT.
            # (nogrid eq 0 means that keyword nogrid was not used, hence
            # minimization is done one the grid...)
            if nogrid == False:
                for j in range(npa * nq):
                    chi2_grid[j] = kinem_fitfunc_ellipse(np.array([pa_grid[j], q_grid[j], x0, y0]),
                                                         nterms=4, r=rad[i], xbar=xbar, ybar=ybar, moment=moment,
                                                         interp=interp,
                                                         even=even, img=img, x0=x0, y0=y0, badpix=badpix,
                                                         er_interp=er_interp, grid=True)
                pa_mpf = pa_grid[np.argmin(chi2_grid)]
                q_mpf = q_grid[np.argmin(chi2_grid)]

            # Perform least-squares minimization of the harmonic coefficients
            # starting from the best values of the global minimization.
            # In case of the EVEN moment minimize a1,b1,a2,b2. In case of
            # the ODD moment minimize a1,a3,b3.
            if img is not None:
                if even:
                    if fixcen:
                        par = np.array([pa_mpf, q_mpf])
                        fa = {'nterms': 4, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'interp': interp,
                              'even': True, 'img': img, 'x0': x0, 'y0': y0, 'badpix': badpix, 'er_interp': er_interp,
                              'mpf': True}
                        sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                        PA_min = sol.params[0]
                        q_min = sol.params[1]
                        x0s = x0
                        y0s = y0
                        er_PA_min = sol.perror[0]
                        er_q_min = sol.perror[1]
                        er_x0s = 0
                        er_y0s = 0
                    else:  # fixcen
                        par = np.array([pa_mpf, q_mpf, x0, y0])
                        fa = {'nterms': 4, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'interp': interp,
                              'even': True, 'img': img, 'x0': x0, 'y0': y0, 'badpix': badpix, 'er_interp': er_interp,
                              'mpf': True}
                        sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                        PA_min = sol.params[0]
                        q_min = sol.params[1]
                        x0s = sol.params[2]
                        y0s = sol.params[3]
                        er_PA_min = sol.perror[0]
                        er_q_min = sol.perror[1]
                        er_x0s = sol.perror[2]
                        er_y0s = sol.perror[3]
                else:  #
                    if fixcen:
                        if q_mpf < 0.2:
                            q_mpf = 0.2
                        par = np.array([pa_mpf, q_mpf])
                        fa = {'nterms': 4, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'interp': interp,
                              'even': even, 'img': img, 'x0': x0, 'y0': y0, 'er_interp': er_interp, 'mpf': True}
                        sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                        PA_min = sol.params[0]
                        q_min = sol.params[1]
                        x0s = x0
                        y0s = y0
                        er_PA_min = sol.perror[0]
                        er_q_min = sol.perror[1]
                        er_x0s = 0
                        er_y0s = 0
                    else:  # fixcen
                        if q_mpf < 0.2:
                            q_mpf = 0.2
                        par = np.array([pa_mpf, q_mpf, x0, y0])
                        fa = {'nterms': 6, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'interp': interp,
                              'even': even, 'img': img, 'x0': x0, 'y0': y0, 'er_interp': er_interp, 'mpf': True,
                              'fixcen': False, 'allterms': True}
                        sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                        PA_min = sol.params[0]
                        q_min = sol.params[1]
                        x0s = sol.params[2]
                        y0s = sol.params[3]
                        er_PA_min = sol.perror[0]
                        er_q_min = sol.perror[1]
                        er_x0s = sol.perror[2]
                        er_y0s = sol.perror[3]
            else:  # img
                if fixcen:
                    par = np.array([pa_mpf, q_mpf])
                    fa = {'nterms': 4, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'x0': x0, 'y0': y0,
                          'interp': interp, 'er_interp': er_interp, 'mpf': True, 'even': even}
                    sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                    PA_min = sol.params[0]
                    q_min = sol.params[1]
                    x0s = x0
                    y0s = y0
                    er_PA_min = sol.perror[0]
                    er_q_min = sol.perror[1]
                    er_x0s = 0
                    er_y0s = 0
                else:
                    par = np.array([pa_mpf, q_mpf, x0, y0])
                    fa = {'nterms': 6, 'r': rad[i], 'xbar': xbar, 'ybar': ybar, 'moment': moment, 'interp': interp,
                          'er_interp': er_interp, 'x0': x0, 'y0': y0, 'mpf': True, 'even': even, 'fixcen': False,
                          'allterms': True}
                    # pdb.set_trace()
                    sol = mpfit(kinem_fitfunc_ellipse, par, parinfo=parinfo, functkw=fa, quiet=1)
                    # pdb.set_trace()
                    PA_min = sol.params[0]
                    q_min = sol.params[1]
                    x0s = sol.params[2]
                    y0s = sol.params[3]
                    er_PA_min = sol.perror[0]
                    er_q_min = sol.perror[1]
                    er_x0s = sol.perror[2]
                    er_y0s = sol.perror[3]
        else:  # PA and Q are set to fixed values: skip all minimization
            #
            # check if PAQ is an array of (nrad*2) values or has just 2 values
            #
            if paq.shape[0] > 2:
                PA_min = paq[k]
                q_min = paq[k + 1]
                k = k + 2
            else:
                PA_min = paq[0]
                q_min = paq[1]

            x0s = x0
            y0s = y0
            er_x0s = 0.
            er_y0s = 0.
            er_PA_min = 0.
            er_q_min = 0.

        if (img is not None) & even:
            coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr, w, er_coeff = kinem_fitfunc_ellipse(
                np.array([PA_min, q_min, x0s, y0s]),
                nterms=ntrm, r=rad[i], xbar=xbar, ybar=ybar, moment=moment, interp=interp,
                even=True, allterms=True, img=img, x0=x0, y0=y0, badpix=badpix,
                er_interp=er_interp)
        else:
            coeff, xell, yell, theta, momEll, er_momEll, momFit, momFitExtr, w, er_coeff, = kinem_fitfunc_ellipse(
                np.array([PA_min, q_min, x0s, y0s]),
                nterms=ntrm, r=rad[i], xbar=xbar, ybar=ybar, moment=moment, interp=interp,
                even=even, allterms=allterms, img=None, x0=x0, y0=y0, vsys=vsys,
                er_interp=er_interp)

            # Stops the fit when there are less than 3/4 of the pixels sampled
        # along the best fitting ellipse. Use COVER keyword to relax this
        # condition (e.g. when also setting PAQ). In case of IMG keyword
        # stop when lenght of the ellipse semi-major axis is 10% larger than
        # the larger side of the image. Stop also if keyword sky is set and the
        # intensity is 0.5xSKY.

        if w[0].size < xell.size * cover:
            print('cover fraction limit reached')
            break
        if sky is not None:
            if coeff[0] < sky:
                print('sky limit reached')
                break
        if (img is not None) and (radius is None):
            if rad[i] > mx_img / 2 + mx_img * 0.1:
                print('edge of the image reached')
                break

        if verbose:
            if i == 0:
                print('       Radius,    RAD,   PA,     Q,    Xcen[pix],  Ycen[pix],   num. of ellipse elements')
                f = open(output_file + "/kinemetry_outputs.txt", "w")
                f.write('       Radius,    RAD,   PA,     Q,    Xcen[pix],  Ycen[pix],   num. of ellipse elements\n')
                f.close()
            print('%3i %11s %5.2f %7.2f %7.3f %7.1f %7.1f %3i' % (
            i, '-th radius  ', rad[i] * scale, PA_min, q_min, x0s, y0s, w[0].size))
            f = open(output_file + "/kinemetry_outputs.txt", "a")
            f.write('%3i %11s %5.2f %7.2f %7.3f %7.1f %7.1f %3i\n' % (
            i, '-th radius  ', rad[i] * scale, PA_min, q_min, x0s, y0s, w[0].size))
            f.close()
        # assining vsys to zeroth term
        if vsys is not None:
            coeff[0] = vsys
            er_coeff[0] = 0.

        pa[i] = PA_min
        q[i] = q_min
        cf[i, :] = coeff

        er_cf[i, :] = er_coeff
        er_pa[i] = er_PA_min
        er_q[i] = er_q_min

        xc[i] = x0s
        yc[i] = y0s
        er_xc[i] = er_x0s
        er_yc[i] = er_y0s

        nelem[i] = w[0].size

        #
        # reconstruction of moments
        #
        xellip = np.concatenate([xellip, xell[w]])
        yellip = np.concatenate([yellip, yell[w]])
        eccano = np.concatenate([eccano, theta[w]])
        ex_mom = np.concatenate([ex_mom, momEll])
        vrec = np.concatenate([vrec, momFit])
        if even:
            vv = np.concatenate([vv, coeff[0] + 0 * coeff[2] * np.cos(theta[w])])
        else:
            vv = np.concatenate([vv, coeff[0] + coeff[2] * np.cos(theta[w])])

        #
        # optional plotting
        #                     
        if plot:
            if even:
                if img is not None:
                    plt.clf()
                    xmin = int(round(x0s)) - 2 * rad[i]
                    xmax = int(round(x0s)) + 2 * rad[i]
                    ymin = int(round(y0s)) - 2 * rad[i]
                    ymax = int(round(y0s)) + 2 * rad[i]
                    if xmax < img.shape[0] and ymax < img.shape[1]:
                        ext = [xmin, xmax, ymin, ymax]
                    else:
                        ext = [np.min(xbin), np.max(xbin), np.min(ybin), np.max(ybin)]
                    peak = img[int(round(x0s)), int(round(y0s))]
                    levels = peak * 10 ** (-0.4 * np.arange(0, 10, 0.5)[::-1])  # 0.5 mag/arcsec^2 steps

                    ax1 = plt.subplot(411)
                    ax1.imshow(np.log10(img), vmin=np.min(np.log10(levels)), vmax=np.max(np.log10(levels)),
                               origin='lower', extent=ext, cmap='gray')
                    ax1.contour(np.log10(img), levels=np.log10(levels), colors='black', linewidths=1, origin='lower',
                                extent=ext)
                    plt.plot(xell * scale, yell * scale, '+')
                    ellipse = Ellipse(xy=(x0s, y0s), width=2 * rad[i] * scale, height=2 * rad[i] * scale * (q_min),
                                      angle=PA_min - 90, edgecolor='r', fc='None', lw=2)
                    plt.plot(x0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.cos(np.radians(PA_min - 90)),
                             y0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.sin(np.radians(PA_min - 90)), c='r',
                             lw=2)
                    ax1.add_patch(ellipse)
                    if name:
                        ax1.set_title(name)
                    pr_rad = round(rad[i] * scale, 2)

                    plt.subplot(412)
                    #                    pdb.set_trace()
                    if (nogrid is False):
                        plt.tricontour(pa_grid, q_grid, chi2_grid, linewidths=0.5, colors='k')
                        plt.tricontourf(pa_grid, q_grid, chi2_grid, cmap="RdBu_r")
                    plt.plot(pa_grid, q_grid, '.', c='k', ms=1)
                    plt.xlabel('PA [deg]', fontweight='bold')
                    plt.ylabel('q', fontweight='bold')
                    plt.title('Grid fit for radius R=' + np.str(pr_rad))
                    plt.tick_params(axis='both', which='both', top=True, right=True)
                    if paq is None:
                        plt.plot(pa_mpf, q_mpf, 'o', c='k')
                    plt.plot(PA_min, q_min, 'o', c='red')

                    plt.subplot(413)
                    plt.errorbar(np.degrees(theta[w]), momEll, yerr=er_momEll, marker='.', ls='none', c='k')
                    plt.plot(np.degrees(theta[w]), momFit, c='red', label='kinemetry fit (all terms)', zorder=10)
                    plt.plot(np.degrees(theta[w]), coeff[0] * (theta[w] * 0 + 1), c='blue', label=r'a$_0$')
                    plt.plot(np.degrees(theta), momFitExtr, c='skyblue', label='kin fit (extraplated)')
                    plt.xlabel(r'$\theta$ [degr]')
                    plt.ylabel('moment')
                    plt.tick_params(axis='both', which='both', top=True, right=True)

                    plt.subplot(414)
                    plt.errorbar(np.degrees(theta[w]), momEll - coeff[0] * (theta[w] * 0 + 1), yerr=er_momEll,
                                 marker='.', ls='none', c='k')
                    plt.plot(np.degrees(theta[w]),
                             coeff[1] * np.sin(theta[w]) + coeff[2] * np.cos(theta[w]) + coeff[3] * np.sin(
                                 2 * theta[w]) + coeff[4] * np.cos(2 * theta[w]), c='red',
                             label=r'$a_\sin(\theta)+b_1\cos(\theta)+a_2\sin(2\theta)+b_2\cos(2\theta)$')
                    plt.plot(np.degrees(theta[w]),
                             coeff[1] * np.sin(theta[w]) + coeff[2] * np.cos(theta[w]) + coeff[3] * np.sin(
                                 2 * theta[w]) + coeff[4] * np.cos(2 * theta[w]) + coeff[7] * np.sin(4 * theta[w]) +
                             coeff[8] * np.cos(4 * theta[w]), c='green',
                             label=r'$a_\sin(\theta)+b_1\cos(\theta)+a_2\sin(2\theta)+b_2\cos(2\theta) + a_4\sin(4\theta)+b_4\cos(4\theta)$')
                    plt.xlabel(r'$\theta$ [degr]')
                    plt.ylabel('moment - a$_0$')
                    plt.tick_params(axis='both', which='both', top=True, right=True)

                    plt.tight_layout()
                    plt.pause(0.01)

                else:
                    plt.clf()
                    ax1 = plt.subplot(411)
                    plot_velfield(xbin, ybin, moment, colorbar=True, label='km/s', nodots=True)
                    plt.plot(xell * scale, yell * scale, '+')
                    ellipse = Ellipse(xy=(x0s, y0s), width=2 * rad[i] * scale, height=2 * rad[i] * scale * (q_min),
                                      angle=PA_min - 90, edgecolor='r', fc='None', lw=2)
                    plt.plot(x0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.cos(np.radians(PA_min - 90)),
                             y0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.sin(np.radians(PA_min - 90)), c='r',
                             lw=2)
                    ax1.add_patch(ellipse)
                    if name:
                        ax1.set_title(name)

                    pr_rad = round(rad[i] * scale, 2)

                    plt.subplot(412)
                    if paq is None:
                        plt.tricontour(pa_grid, q_grid, chi2_grid, linewidths=0.5, colors='k')
                        plt.tricontourf(pa_grid, q_grid, chi2_grid, cmap="RdBu_r")
                    plt.plot(pa_grid, q_grid, '.', c='k', ms=1)
                    plt.xlabel('PA [deg]', fontweight='bold')
                    plt.ylabel('q', fontweight='bold')
                    plt.title('Grid fit for radius=' + np.str(pr_rad))
                    plt.tick_params(axis='both', which='both', top=True, right=True)
                    if paq is None:
                        plt.plot(pa_mpf, q_mpf, 'o', c='k')
                    plt.plot(PA_min, q_min, 'o', c='red')

                    plt.subplot(413)
                    plt.errorbar(np.degrees(theta[w]), momEll, yerr=er_momEll, marker='.', ls='none', c='k')
                    plt.plot(np.degrees(theta[w]), momFit, c='red', label='kinemetry fit (all terms)', zorder=10)
                    plt.plot(np.degrees(theta[w]), coeff[0] * (theta[w] * 0 + 1), c='blue', label=r'a$_0$')
                    plt.plot(np.degrees(theta), momFitExtr, c='skyblue', label='kin fit (extraplated)')
                    plt.xlabel(r'$\theta$ [degr]')
                    plt.ylabel('moment')
                    plt.tick_params(axis='both', which='both', top=True, right=True)
                    plt.legend(loc='best')

                    plt.subplot(414)
                    plt.errorbar(np.degrees(theta[w]), momEll - coeff[0] * (theta[w] * 0 + 1), yerr=er_momEll,
                                 marker='.', ls='none', c='k')
                    plt.plot(np.degrees(theta[w]),
                             coeff[1] * np.sin(theta[w]) + coeff[2] * np.cos(theta[w]) + coeff[3] * np.sin(
                                 2 * theta[w]) + coeff[4] * np.cos(2 * theta[w]), c='red',
                             label=r'$a_\sin(\theta)+b_1\cos(\theta)+a_2\sin(2\theta)+b_2\cos(2\theta)$')
                    plt.plot(np.degrees(theta[w]),
                             coeff[1] * np.sin(theta[w]) + coeff[2] * np.cos(theta[w]) + coeff[3] * np.sin(
                                 2 * theta[w]) + coeff[4] * np.cos(2 * theta[w]) + coeff[7] * np.sin(4 * theta[w]) +
                             coeff[8] * np.cos(4 * theta[w]), c='green',
                             label=r'$a_\sin(\theta)+b_1\cos(\theta)+a_2\sin(2\theta)+b_2\cos(2\theta) + a_4\sin(4\theta)+b_4\cos(4\theta)$')
                    #                    plt.plot(np.degrees(theta), momFitExtr, c='blue')
                    plt.xlabel(r'$\theta$ [degr]')
                    plt.ylabel('moment - a$_0$')
                    plt.tick_params(axis='both', which='both', top=True, right=True)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.pause(0.01)

            else:
                plt.clf()
                ax1 = plt.subplot(411)
                plot_velfield(xbin, ybin, moment, colorbar=True, label='km/s', nodots=True)
                plt.plot(xell * scale, yell * scale, '+')
                ellipse = Ellipse(xy=(x0s, y0s), width=2 * rad[i] * scale, height=2 * rad[i] * scale * (q_min),
                                  angle=PA_min - 90, edgecolor='r', fc='None', lw=2)
                plt.plot(x0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.cos(np.radians(PA_min - 90)),
                         y0s + np.array([-rad[i] * scale, rad[i] * scale]) * np.sin(np.radians(PA_min - 90)), c='r',
                         lw=2)
                ax1.add_patch(ellipse)
                if name:
                    ax1.set_title(name)

                pr_rad = round(rad[i] * scale, 2)

                plt.subplot(412)
                if paq is None:
                    plt.tricontour(pa_grid, q_grid, chi2_grid, linewidths=0.5, colors='k')
                    plt.tricontourf(pa_grid, q_grid, chi2_grid, cmap="RdBu_r")
                plt.plot(pa_grid, q_grid, '.', c='k', ms=1)
                plt.xlabel('PA [deg]', fontweight='bold')
                plt.ylabel('q', fontweight='bold')
                plt.title('Grid fit for radius R=' + np.str(pr_rad))
                plt.tick_params(axis='both', which='both', top=True, right=True)
                if paq is None:
                    plt.plot(pa_mpf, q_mpf, 'o', c='k')
                plt.plot(PA_min, q_min, 'o', c='red')

                plt.subplot(413)
                plt.errorbar(np.degrees(theta[w]), momEll, yerr=er_momEll, marker='.', ls='none', c='k')
                plt.plot(np.degrees(theta[w]), momFit, c='red', label='kinemetry fit (all terms)', zorder=10)
                plt.plot(np.degrees(theta[w]), coeff[0] + coeff[2] * np.cos(theta[w]), c='blue',
                         label=r'$a_0+b_1 \cos(\theta)$')
                plt.plot(np.degrees(theta), momFitExtr, c='skyblue', label='kin fit (extrapolated)')
                plt.xlabel(r'$\theta$ [degr]')
                plt.ylabel(r'velocity V')
                plt.tick_params(axis='both', which='both', top=True, right=True)
                plt.legend(loc='best')

                plt.subplot(414)
                plt.errorbar(np.degrees(theta[w]), momEll - coeff[0] - coeff[2] * np.cos(theta[w]), yerr=er_momEll,
                             marker='.', ls='none', c='k')
                plt.plot(np.degrees(theta[w]),
                         coeff[1] * np.sin(theta[w]) + coeff[3] * np.sin(3 * theta[w]) + coeff[4] * np.cos(
                             3 * theta[w]), c='red', label=r'$a_1 \sin(\theta) + a_3 \sin(\theta) + b_3 \cos(\theta)$')
                plt.plot(np.degrees(theta[w]),
                         coeff[1] * np.sin(theta[w]) + coeff[3] * np.sin(3 * theta[w]) + coeff[4] * np.cos(
                             3 * theta[w]) + coeff[5] * np.sin(5 * theta[w]) + coeff[6] * np.cos(5 * theta[w]),
                         c='green',
                         label=r'$a_1 \sin(\theta) + a_3 \sin(3\theta) + b_3 \cos(3\theta)+ a_5 \sin(5\theta) + b_5 \cos(5\theta)$')
                plt.xlabel(r'$\theta$ [degr]')
                plt.ylabel(r'$V - V_{sys}-V_{rot}(R)\cos(\theta)$')
                plt.tick_params(axis='both', which='both', top=True, right=True)
                plt.legend(loc='best')

                plt.savefig(fname=output_file + "/" + str(i) + ".png")
                plt.tight_layout()
                plt.pause(0.01)

    # END OF MAIN LOOP
    #
    # Final outputs (back to physical scale (arcsec))
    #
    wz = np.where(q != 0)  # remove unused array elements

    xellip = xellip * scale
    yellip = yellip * scale

    rad = rad[wz] * scale
    pa = pa[wz]
    q = q[wz]
    cf = cf[wz[0], :]
    er_cf = er_cf[wz[0], :]
    er_pa = er_pa[wz]
    er_q = er_q[wz]
    er_xc = er_xc[wz] * scale
    er_yc = er_yc[wz] * scale
    xc = xc[wz] * scale
    yc = yc[wz] * scale

    # exclude the central pixel in the ring case
    if ring is not None:
        xellip = xellip[1:]
        yellip = yellip[1:]
        vv = vv[1:]
        vrec = vrec[1:]
        eccano = eccano[1:]
        ex_mom = ex_mom[1:]

    #
    # calculation of the circular velocity (not for the photometry mode):
    # (VELCIRC and VELKIN are calculated above before plotting and final outputs)
    #   - fixed PA and q
    #   - using only cf[*,2] terms (cosine terms)
    #   - xellipF and yellipF are new ellipes along which gascirc is calculated
    #     Each ellipse has 100 points - they are different from xellip,yellip
    #
    qfix = np.median(q[1:-1])
    PAfix = np.radians(np.median(pa[1:-1]))
    xellipF = np.array([])
    yellipF = np.array([])
    vvF = np.array([])
    for i in range(rad.size):
        theta = kin_range(0.0, 2.0 * np.pi, 100)
        xF = rad[i] * np.cos(theta)
        yF = rad[i] * np.sin(theta) * qfix
        xEllF = xF * np.cos(PAfix) - yF * np.sin(PAfix) + x0s
        yEllF = xF * np.sin(PAfix) + yF * np.cos(PAfix) + y0s
        xellipF = np.concatenate([xellipF, xEllF])
        yellipF = np.concatenate([yellipF, yEllF])
        vvF = np.concatenate([vvF, cf[i, 2] * np.cos(theta)])

    #
    # reconstruction of kinematic moment maps
    #
    if bmodel is True:
        if img is not None:
            #            xout=np.arange(img.shape[0])
            #            yout=np.arange(img.shape[1])
            #            velcirc = griddata((xellip, yellip), vv, (xout[None,:], yout[:,None]), method='linear')
            #            velkin = griddata((xellip, yellip), vrec, (xout[None,:], yout[:,None]), method='linear')
            #            gascirc = griddata((xellipF, yellipF), vvF, (xout[None,:], yout[:,None]), method='linear')
            velcirc = griddata((xellip, yellip), vv, (xx, yy), method='linear')
            velkin = griddata((xellip, yellip), vrec, (xx, yy), method='linear')
            gascirc = griddata((xellipF, yellipF), vvF, (xx, yy), method='linear')

        else:
            velcirc = kinem_trigrid_irregular(xellip, yellip, vv, xbin, ybin, missing=123456789)  # circular velocity
            velkin = kinem_trigrid_irregular(xellip, yellip, vrec, xbin, ybin,
                                             missing=123456789)  # full kinemetry reconstruction
            gascirc = kinem_trigrid_irregular(xellipF, yellipF, vvF, xbin, ybin,
                                              missing=123456789)  # "gas circular velocity", same as velcirc, but for fixed Pa and Q
    else:
        velcirc = vv
        velkin = vrec
        gascirc = vvF  # disabled as its ellipse (xellipF, yellipF) parameters are not the same the main outouts (xellip, yellip)

    # ==========================================================================
    # adding relevant arrays into structure
    # ==========================================================================
    results.rad = rad  # radii for kinemetric analysis
    results.pa = pa  # ellipse position angle
    results.q = q  # ellipse flattening
    results.cf = cf  # harmonic expansion terms
    results.xc = xc  # x centre
    results.yc = yc  # y centre
    results.er_cf = er_cf  # errors on harmonic expansion terms
    results.er_q = er_q  # error on ellipse flattening
    results.er_pa = er_pa  # error on ellipse position angle
    results.er_xc = er_xc  # error on the centre location
    results.er_yc = er_yc  # error on the centre location
    results.velcirc = velcirc  # reconstructed "circular velocity" (a0 + b1*cos(theta)) map (for xbin and ybin of the input map)
    results.velkin = velkin  # reconstructed kinemetry map using nterms (for xbin and ybin of the input map)
    results.gascirc = gascirc  # reconstructed "gas" circular velocity (b1*cos(theta) map when PA and Q are fixed (for xbin and ybin of the input map)
    results.vsys = vsys  # systemic velocity (a0 term)
    results.Xellip = xellip  # x- coordiantes of the best fitting ellipses
    results.Yellip = yellip  # y- coordiantes of the best ellipses
    results.XellipC = xellipF  # x- coordiantes of the ellipses with fixed PA and Q, used for gascirc
    results.YellipC = yellipF  # y- coordiantes of the ellipses with fixed PA and Q, used for gascirc
    results.eccano = eccano  # eccentric anomaly of the best ellipses
    results.ex_mom = ex_mom  # extracted "data" along the best fit ellispes
    results.Nelem = nelem  # number of eccentric anomaly points for each extracted ellipse
    results.vv = vv  # circular velocity along the best fitting ellipse
    results.vrec = vrec  # full kinemetry reconstruction along the best fitting ellipse
    results.vvF = vvF  # "gas" circular velocity along the ellipse
    results.er_momEll = er_momEll  # "gas" circular velocity along the ellipse
    if sky:
        results.sky = sky  # passing the value of the sky keyword
    if ring:
        results.ring = ring  # passing the value of the ring keyword

    #    pdb.set_trace()
    return results

# ----------------------------------------------------------------------------

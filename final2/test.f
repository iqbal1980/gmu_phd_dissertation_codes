
Fortran Code for Model

Main Program

      implicit double precision (a-h,o-z)
      dimension vnfat(14),vnfkb(19),vjnk(23)

c------ T cell model--------------------------------------
c**By Pei-Chi Yang
c**Copyright 2007 Saleet Jafri Lab. All rights reserved.
c**9-10-2007 
c---------------------------------------------------------


c define parameter values

c For fchan & fpump      
c      v1 = 1800.0 
      v1 = 90.0 
      ak3 = 0.1
      v2 = 0.00
      v3 = 1.0
	  
c For IP3R	  
      a1 = 400.0
      a3 = 400.0
      a4 = 0.2
      d1 = 0.13
      d3 = 0.9434
      d4 = 0.1445
      a2 = 0.2
      a5 = 20.0 
      d2 = 1.049
      d5 = 82.34e-3
      c2 = 0.2 
	  
c For buffer
      Bscyt = 225.0
      aKscyt = 0.1
      Bscas = 5000.0
      aKscas = 0.1
      Bm = 111.0
      aKm = 0.123
      Bser = 2000.0
      aKser = 1.0
c Univeral constants
      F = 96500.0
      T = 310.0
      R = 8314.0
      z = 2.0
      Cm = 2.5e-3               !nF
c parameters for Ion channels
      gna = 3.0					!ns for I_Na
      gt = 3.0            !ns  for I_K
      Ek = -84.0                !mV  for I_K
      Eca = 60.0                !mV for I_b_Ca   
      Emax = 0.8             !ns for I_K(Ca)
      akd = 0.45                !uM for I_K(Ca)
      ax = 3.0
      gtrpc3 = 0.00195				!for membrane leak
      aKmca = 0.6                 !uM for I_Ca_L
      PcaL = 0.3e-7             !cm^3/s !L-type Ca2+

c parameters for I_crac
      taoinact = 40.0           ! (s) inaction for I_crac
      taoinact = 100.0           ! (s) inaction for I_crac
      taoact = 3.0              ! (s) activation for I_crac
      Pca = 2.8d-10             !cm^3/s !CRAC
      alphi = 1.0
      alpho = 0.341
      akk = 200.0                !uM
      akj = 1.0                 !uM
	  
c parameters for PMCA	  
      vu = 1.0*1540000.0            !ions
      vm = 1.0*2200000.0            !ions
      aku = 0.303
      akmp = 0.14
      aru = 1.8
      arm = 2.1

c parameters for calculating [IP3]
      V2ip3 = 12.5
      ak2ip3 = 6.0
      V3ip3 = 0.9
      ak3ip3 = 0.1
      ak4ip3 = 1.0

	
      taos = 0.5 !diffusion constant in cytosol

c parameters for mitochondria
      psi = 160.0               !mV
      alphm = 0.2
      Pmito = 2.776e-14    !1/s
      Vnc = 1.836          !uM/s     
      aNa = 5000.0               !uM
      akna = 8000.0             !uM
      akca = 8.0                !uM
      
c Cell Geometry
      Vt = 2.0e-12              !L T cell volume     
      Vcyto = Vt*0.55           !cytoplasm volume
      Ver = Vcyto*0.2          !ER volume
      Vmito = Vcyto*0.08        !mito volume 
      Vss = Vcyto*0.1           !subspace volume
      c1 = Ver/Vcyto            !ER-cytosol ratio
      c3 = Vmito/Vss            !mito-subspace ratio
      c4 = Vcyto/Vss
      c5 = Vmito/Vcyto
          
      b1 = a1*d1
      b2 = a2*d2
      b3 = a3*d3
      b4 = a4*d4
      b5 = a5*d5
c     initial conditions      
      iflag = 1
      if (iflag.eq.1) then 
         open (unit = 13,file='restart.init5')
	 read (13,*) cai,caer,cam,aip3,dag,x100,x110,dpmca,cas,
     +               V,an,aj,ad,af,aa5,ah5,xCai,vjnk(23),vnfat(6),
     +               vnfat(9),vnfkb(1),vnfkb(8),proinact,proact,
     +               dpmca,x000,x010,x001,x101,x011,x111
         close(13)
      else         
         cai = 0.1
         caer = 300.0
         cam = 0.1
         aip3 = 0.01
         dag = 0.01
         x100 = 0.1
         x110 = 0.1
         dpmca = 1.0
         cas = 0.1
         V = -70.0
         an = 0.02
         aj = 1.0
         ad = 0.0
         af = 1.0
         aa5 = 0.0
         ah5 = 1.0
         anfatc = 6.14972E-06      ! [NFAT:C*]c (uM)
         anfatn = 9.47710E-04       ! [NFAT:C*]n (uM)
         nfkbc = 7.31124E-02		!nM
         proinact = 1.0
         proact = 0.0
         x000 = 0.0
         x010 = 0.0
         x001 = 0.0
         x101 = 0.0
         x011 = 0.0
         x111 = 0.0
         u4 = 0.0
         u5 = 0.0
 
      endif
      
      currca = 0.0
      procrac = 0.0
      B = 0.0
      bb = 0.0
      cJuni = 0.0
      cJnc = 0.0
      cJcrac = 0.0
      cJpmca = 0.0
      Eca = 60.0                !mV
      fchan = 8.73055d0
      fpump = 8.73057d0
      fleak = 0.0
      aninf = 0.0
      ajinf = 1.0
      afinf = 1.0
      adinf = 0.0
      taoan = 0.0
      taoaj = 0.0
      taoad = 0.0
      taoaf = 0.0
      cdinf = 0.0
      cfinf = 1.0
      afca = 0.0
      barIca = 0.0
      aIca = 0.0
      aIt = 0.0
      aIkca = 0.0  
      aIna = 0.0
      aItrpc3 = 0.0
      aItrpm4 = 0.0
      aIcl = 0.0
      cJca = 0.0
      cJtrpc3 = 0.0
      aIpmca = 0.0
      aa1 = 0.0
      aa2 = 0.0
      aa3 =0.0
      aa4 =0.0
      
      ah1 = 0.0
      ah2 = 0.0
      ah3 = 0.0
      ah4 = 0.0
   
      dt = 0.0001                 ! integration time step (s)
      time = 0.0

      write (8,100) time,cai,caer,cam,cas
      write (9,100) time,fchan,fpump,x100,x110
      write (10,100) time,fleak,aip3,cJuni,cJnc
      write (11,100) time,currca,procrac,cJcrac
      write (12,100) time,proact,proinact,cfinf,cdinf
      write (14,100) time,u4,u5,dpmca,cJpmca     
      write (15,100) time,aIca,aIt,aIkca,aIna
      write (16,100) time,aninf,ajinf,taoan,taoaj
      write (17,100) time,an,aj,ad,af
      write (18,100) time,adinf,afinf,taoad,taoaf
      write (19,100) time,V,barIca,afca
      write (20,100) time,aa3,aa4,ah3,ah4
      write (21,100) time,aa5,ah5,cJca,aIpmca
      write (22,100) time,aItrpc3,cJtrpc3,Eca,dag
      write (23,100) time,aItrpm4,aIcl
      write (24,100) time,vjnk(23)
      write (25,100) time,vnfat(9),vnfat(6)
      write (26,100) time,vnfkb(1),vnfkb(8)

      
c     Start integration loop
      do j=1,2400		!1000 for fig 3&4, 1200 for fig 7
         do i=1,10000
            time = time + dt
            u2 = cai*cai
              
c for figure 3 simulation 			  
c            if (time.ge.420.0.and.time.lt.800.0) then
c               cao = 2000.0
c            else
c               cao = 0.0
c            endif
c for figure 4 simulation 
c            if (time.ge.500.0.and.time.lt.515.0) then !fig2&4
c               cao = 2000.0
c            elseif (time.ge.860.0.and.time.lt.900.0) then !fig4
c               cao = 2000.0
c            else
c               cao = 0.0        
c            endif

c Calculate IP3 concentration for Fig 3 and 4 
            if (time.lt.64.0) then
               prodip3 = 0.05
            else
               prodip3 = 5.0
            endif            

c for figure 7-9
              if (time.lt.1200.0) then
                cao = 2000.0
              else
                cao = 0.0
              endif
              if (time.lt.200.0.or.time.gt.1200) then
                prodip3 = 0.01
              else
                prodip3 = 5.0
              endif

c******** Calculate IP3 concentration **********
            aip3 = aip3 + dt*(prodip3-V2ip3/
     +           (1.0+(ak2ip3/aip3))-(V3ip3/(1+(ak3ip3/aip3)))*
     +           (1.0/(1.0+(ak4ip3/cai))))
            dag = dag + dt*(prodip3-1.d0*dag)

c************  buffer  **********
          
            buer = 1.0/(1.0+Bser*aKser/((aKser+caer)**2)+
     +           (Bm*aKm)/((aKm+caer)**2))

            bucyt = 1.0/(1+(Bscyt*aKscyt)/((aKscyt+cai)**2)+ 
     +           (Bm*aKm)/((aKm+cai)**2))

            bucas = 1.0/(1+(Bscas*aKscas)/((aKscas+cas)**2)+ 
     +           (Bm*aKm)/((aKm+cas)**2))
            
c******** IP3R *************
            fchan = c1*(v1*(x110**3))*(caer - cai)
            fleak = c1*v2*(caer - cai)
            fpump = v3*u2/(ak3*ak3 + u2)  
            
c******* mitochondrial ********
           
            if (psi == 0) then
               cJuni = (Pmito/Vmito)*(alphm*cam-alphi*cas)  
            else
               bb = (z*psi*F)/(R*T)
               cJuni=(Pmito/Vmito)*bb*((alphm*cam*exp(-bb)-
     +              alphi*cas)/(exp(-bb)-1))
            endif        
            som = aNa**3*cam/(akna**3*akca)
            soe = aNa**3*cas/(akna**3*akca)
            cJnc = Vnc*(exp(0.5*psi*F/(R*T))*som-
     +           exp(-0.5*psi*F/(R*T))*soe/
     +           (1+aNa**3/(akna**3)+cam/akca+som+
     +           aNa**3/(akna**3)+cas/akca+soe))
            cam = cam + dt*0.01*(cJuni-cJnc)

c----------------------------
c current calculation         
c----------------------------
        
c  (1)*** Icrac ***
			if (V == 0) then
               currca = Pca*z*F*(alphi*cas-alpho*cao)  
            else
               B = (z*V*F)/(R*T)
               currca = Pca*z*F*B*((alphi*cas*exp(B)-alpho*cao)/
     +              (exp(B)-1))
            endif 
           cfinf = (akj**2/(akj**2+cas**2))
           cdinf = (akk**4.7/(akk**4.7+caer**4.7))	
c !Nature Communicationsvolume 9, Article number: 4536 (2018
           taoinact = 200*(2.0**2/(2.0*2+cas**2))
           proinact = proinact + dt*(cfinf-proinact)/taoinact
           proact = proact + dt*(cdinf-proact)/taoact    
           procrac = proact*proinact
           currca = currca*procrac
           cJcrac = -currca/(1e6*Vss*z*F) !convert pA/s to uM/s
           
		   
c  (2)*** L-type ca2+ ***    
			if (V == 0) then
               barIca = PcaL*z*F*(alphi*cai-alpho*cao)  
            else
               B = (z*V*F)/(R*T)
               barIca = PcaL*z*F*B*((alphi*cai*dexp(B)-alpho*cao)/
     +              (dexp(B)-1.0))
            endif        
           afca = 1.0/(1.0+(cai/aKmca)**2)
           adinf = 1.0/(1.0+exp(-(V+10.0)/6.24))
           afinf =1.0/(1.0+exp((V+35.06)/8.6))+0.6/(1+exp((50-V)/20.0))
           taoad = adinf*(1.0-exp(-(V+10.0)/6.24))/(0.035*(V+10.0))
           taoaf = 1.0/(0.0197*exp(-(0.0337*(V+10.0))**2)+0.02)
           ad = ad + dt*(adinf-ad)/taoad
           af = af + dt*(afinf-af)/taoaf
           aIca = ad*af*afca*barIca
           cJca = -aIca/(1e6*Vcyto*z*F) !convert to pA/s to uM/s

c  (3)*** K+ channel ***
           aninf = 1.0/(1.0+exp((V+11.0)/(-15.2)))
           ajinf = 1.0/(1.0+exp((V+45.0)/9.0))
           taoan = 1.0/(0.2*exp(0.032*(V+1.42)))
           taoaj = 15.0/(0.03*(exp(0.0083*(V+40.8))+
     +          0.4865*exp(-0.06*(V+60.49))))                      
           an = an + dt*(aninf-an)/taoan
           aj = aj + dt*(ajinf-aj)/taoaj
           aIt = gt*an*aj*(V-Ek)
           
c  (4)*** Ca2+ activated K+
           E = Emax/(1+(akd/cai)**ax)
           aIkca = E*(V-Ek)
                     
c  (5)*** Na+ current *****

c    -------Na activation m-----------
c        
         aa3 = 1.0/(1.0+exp((V+51.9005)/(-3.1011)))
         aa4 = 0.3959*exp(-0.017*V)
         aa5 = aa5 + dt*((aa3-aa5)/aa4)
        
c-----------Na inactivation h
c       
         ah3 = 1.0/(1.0+exp((V+76.5769)/7.1485))
         ah4 = 2.9344*exp(-0.0077*V)
         ah5 = ah5 + dt*((ah3-ah5)/ah4)
c------------INa
         Ena = 30.0   !(R*T/F)*log(Nao/5.0)
         aIna = gna*aa5**3*ah5*(V-Ena)

c******* PMCA ************           
           u4 = (vu*(cai**aru)/(cai**aru+aku**aru))/
     +          (6.6253*1e5) !convert ions/s to uM/s
           u5 = (vm*(cai**arm)/(cai**arm+akmp**arm))/
     +          (6.6253*1e5) !convert ions/s to uM/s           
           w1 = 0.1*cai
           w2 = 0.01
           taom = 1/(w1+w2)
           dpmcainf = w2/(w1+w2)
           dpmca = dpmca + dt*((dpmcainf-dpmca)/taom)           
           cJpmca = (dpmca*u4 + (1-dpmca)*u5)

           aIpmca = cJpmca*z*F*Vcyto*1e6 ! convert to flux (uM/s) to current pA/s
   
c***** TRPM4 channel ****
        Etrpm4 = 0.2d0          !mV
        gtrpm4 = 1.2		            !nS
        xCai_inf = 1.d0/(1.d0+(Cai/1.3)**(-1.1))
        xv = 0.05d0 + (0.95d0/(1.0d0+dexp(-(V+40.0)/15.0)))
        taoxcai = 30.0d0
        axC = xCai_inf/taoxcai
        bxC = (1.d0-xCai_inf)/taoxcai
        xCai = xCai + dt*(axC*(1.d0-xCai)-bxC*xCai)
        aItrpm4 = gtrpm4*xCai*xv*(V-Etrpm4)
        gNatrpm4 = aItrpm4*(Etrpm4-Ena)/
     +         ((V-Ena)*(Ek-Etrpm4)+(V-Ek)*(Etrpm4-Ena))
        gKtrpm4 = aItrpm4*(-Etrpm4+Ek)/
     +         ((V-Ena)*(Ek-Etrpm4)+(V-Ek)*(Etrpm4-Ena))
        aINatrpm4 = gNatrpm4*(V-Ena)
        aIKtrpm4 = gKtrpm4*(V-Ek)

c***** TRPC3 *****
	   if (cao.eq.0) then
	      aItrpc3 = 0.0
           else
              Eca = (R*T/(z*F))*dlog(cao/cai)
              aItrpc3 = gtrpc3*(dag/(dag+2.d0))*(V-Eca)
	   endif
           cJtrpc3 = -aItrpc3/(1e6*Vcyto*z*F) ! convert to uM/s

c***** background Ca current Cab *****
           gcab = 0.0		! NO CaB
           aIcab = gcab*(V-Eca)
           cJcab = -aIcab/(1e6*Vcyto*z*F) ! convert to uM/s

c***** Cl channel *****
        Ecl = -33.d0
        gcl = 70d-3
        aIcl = gcl*(V-Ecl)

c******* membrane ********         

           V = V + dt*(-aIca-aIt-aIkca-currca-aIpmca-aIna
     +                -aItrpc3-aItrpm4-aIcl)/Cm

           
c******** caer/cas/cai ******
           caer = caer + dt*(buer*(fpump-fchan-fleak)/c1)   
            
           cas = cas + dt*bucas*(c3*(cJnc-cJuni)+c4*cJcrac-
     +          c4*(cas-cai)/taos)             
           cai = cai+dt*bucyt*(fchan+fleak-fpump-cJpmca+
     +          (cas-cai)/taos + cJca + cJtrpc3 + cJcab)

           
c     states using implicit euler


           f1 = b5*x010-a5*cai*x000
           f2 = b1*x100-a1*aip3*x000
           f3 = b4*x001-a4*cai*x000
           f4 = b5*x110-a5*cai*x100
           f5 = b2*x101-a2*cai*x100
           f6 = b1*x110-a1*aip3*x010
           f7 = b4*x011-a4*cai*x010
           f8 = b5*x011-a5*cai*x001
           f9 = b3*x101-a3*aip3*x001
           f10 = b2*x111-a2*cai*x110
           f11 = b5*x111-a5*cai*x101
           f12 = b3*x111-a3*aip3*x011
c          x000 = x000 + dt*(f1+f2+f3)
           x000 = 1.d0-(x100+x010+x001+x110+x101+x011+x111)
           x100 = x100 + dt*(f4+f5-f2)
           x010 = x010 + dt*(-f1+f6+f7)
           x001 = x001 + dt*(f8-f3+f9)
           x110 = x110 + dt*(-f4-f6+f10)
           x101 = x101 + dt*(f11-f9-f5)
           x011 = x011 + dt*(-f8-f7+f12)
           x111 = x111 + dt*(-f11-f12-f10)  


           vjnk(13) = cai*1000.d0       ! Ca in nucleus convert uM to nM
           vjnk(14) = cai*1000.d0       ! Ca in cytoplasm convert uM to nM
           call jnk(time,dt,vjnk)

           vnfat(13) = cai      ! Ca in nucleus convert uM to nM
           vnfat(14) = cai      ! Ca in cytoplasm convert uM to nM
           call nfat(time,dt,vnfat)

           vnfkb(13) = cai*1000.d0      ! Ca in nucleus convert uM to nM
           vnfkb(14) = cai*1000.d0      ! Ca in cytoplasm convert uM to nM
           call nfkb(time,10*dt,vnfkb)

      enddo
     
      write (8,100) time,cai,caer,cam,cas
      write (9,100) time,fchan,fpump,x100,x110
      write (10,100) time,fleak,aip3,cJuni,cJnc
      write (11,100) time,currca,procrac,cJcrac
      write (12,100) time,proact,proinact,cfinf,cdinf
      write (14,100) time,u4,u5,dpmca,cJpmca     
      write (15,100) time,aIca,aIt,aIkca,aIna
      write (16,100) time,aninf,ajinf,taoan,taoaj
      write (17,100) time,an,aj,ad,af
      write (18,100) time,adinf,afinf,taoad,taoaf
      write (19,100) time,V,barIca,afca
      write (20,100) time,aa3,aa4,ah3,ah4
      write (21,100) time,aa5,ah5,cJca,aIpmca
      write (22,100) time,aItrpc3,cJtrpc3,Eca,dag
      write (23,100) time,aItrpm4,aIcl,xCai
      write (24,100) time,vjnk(23)
      write (25,100) time,vnfat(9),vnfat(6)
      write (26,100) time,vnfkb(1),vnfkb(8)

      enddo

      open (unit = 13,file='restart.dat')
	 write (13,*) cai,caer,cam,aip3,dag,x100,x110,dpmca,cas,
     +               V,an,aj,ad,af,aa5,ah5,xCai,vjnk(23),vnfat(6),
     +               vnfat(9),vnfkb(1),vnfkb(8),proinact,proact,
     +               dpmca,x000,x010,x001,x101,x011,x111
      close(8)
      close(9)
      close(10)
      close(11)
      close(12)
      close(13)
      close(14)
 100  format (6(2x,1pe12.5))
      stop
      end 

NFAT subroutine

      subroutine nfat(time,dt,vnfat) 
      implicit double precision (a-h,o-z)
      dimension vnfat(14)

c------ T cell model--------------------------------------
c**By Pei-Chi Yang
c**Copyright 2007 Saleet Jafri Lab. All rights reserved.
c**9-10-2007 
c---------------------------------------------------------

c********** NFAT *************
c define parameter values
      ak1 = 0.0000256           ! dephosphorylation of NFAT without assistance of calcineurin 
      ak2 = 0.00256             ! phosphorylation of free NFAT 
      ak3 = 0.005               ! import of phosphorylated NFAT to nucleus
      ak4 = 0.5                 ! export of phosphorylated NFAT from nucleus
      ak5 = 0.0019              ! import of activated calcineurin to nucleus
      ak6 = 0.00092             ! export of activated calcineurin from nucleus
      ak7 = 0.005               ! import of NFAT:Pi:C to nucleus
      ak8 = 0.5                 ! export of NFAT:Pi:C from nucleus
      ak9 = 0.5                 ! import of NFAT:C to nucleus
      ak10 = 0.005              ! export of NFAT:C from nucleus
      ak11 = 6.63            ! association of NFAT:Pi and activated calcineurin
      ak12 = 0.00168            ! dissociation of NFAT:Pi and activated calcineurin
      ak13 = 0.5                ! calcineurin-mediated dephosphorylation of NFAT:Pi:C
      ak14 = 0.00256            ! phosphorylation of NFAT:C
      ak15 = 0.00168            ! dissociation of NFAT:C to NFAT and activated calcineurin
      ak16 = 6.63            ! association of NFAT and activated calcineurin
      ak17 = 0.0015             ! import rate of NFAT to nucleus
      ak18 = 0.00096            ! export rate of NFAT from nucleus
      ak19 = 1.0        ! association of inactive calcineurin with 3 free calcium ions
      ak20 = 1.0                ! dissociation of actived calcineurin to inactive calcineurin and 3 free calcium ions
      ak21 = 0.21               ! import of free calcium ions to nucleus
      ak22 = 0.5                ! export of free calcium ions from nucleus

c Import variables form main conditions
      x1 = vnfat(1)       ! [NFAT]c (uM)
      x2 = vnfat(2)      ! [NFAT]n (uM)
      x3 = vnfat(3)      ! [C*]n (uM)
      x4 = vnfat(4)      ! [C*]c (uM)
      x5 = vnfat(5)     ! [NFAT:Pi]n (uM)
      x6 = vnfat(6)       ! [NFAT:Pi]c (uM)
      x7 = vnfat(7)       ! [NFAT:Pi:C*]n (uM)
      x8 = vnfat(8)      ! [NFAT:Pi:C*]c (uM)
      x9 = vnfat(9)      ! [NFAT:C*]n (uM)
      x10 = vnfat(10)     ! [NFAT:C*]c (uM)
      x11 = vnfat(11)     ! [C]n (uM)
      x12 = vnfat(12)     ! [C]c (uM)
      x13 = vnfat(13)     ! [Ca]n (uM)
      x14 = vnfat(14)     ! [Ca]c (uM)
      
c Initial conditions
      if (time.lt.dt*2.) then
c steady state values for ca = 0.1 uM
      x1 = 5.21909E-04       ! [NFAT]n (uM)
      x2 = 1.10101E-04       ! [NFAT]c (uM)
      x3 = 5.05386E-05       ! [C*]n (uM)
      x4 = 9.14784E-06       ! [C*]c (uM)
      x5 = 2.27232E-04       ! [NFAT:Pi]n (uM)
      x6 = 9.43972E-03       ! [NFAT:Pi]c (uM)
      x7 = 2.52431E-06       ! [NFAT:Pi:C*]n (uM)
      x8 = 2.20743E-06       ! [NFAT:Pi:C*]c (uM)
      x9 = 9.47710E-04       ! [NFAT:C*]n (uM)
      x10 = 6.14972E-06      ! [NFAT:C*]c (uM)
      x11 = 4.91984E-02      ! [C]n (uM)
      x12 = 9.71085E-03      ! [C]c (uM)
      x13 = vnfat(13)     ! [Ca]n (uM)
      x14 = vnfat(14)     ! [Ca]c (uM)
      endif

      volc = 269                ! cytosolic volume (um^3)
      voln = 113                ! nuclear volume (um^3)
      vratio = volc/voln
      vratioc = voln/volc
      
	  
c  number of NFAT molecules in nucleus
      anum_NFAT = (x1+x5+x7+x9)*voln*6.023e2
c  number of NFAT molecules in cytoplasm
      cnum_NFAT = (x2+x6+x8+x10)*volc*6.023e2
c  total number of NFAT molecules
      total_NFAT = (x2+x6+x8+x10)*volc*6.023e2 +
     +     (x1+x5+x7+x9)*voln*6.023e2
c  fraction of NFAT in transcriptionally active form
         fra_NFAT = (x9*voln*6.023e2)/((x2+x6+x8+x10)*volc*6.023e2 +
     +        (x1+x5+x7+x9)*voln*6.023e2)
               
 
c  number of calcineurin molecules in nucleus
      anum_Cn = (x3+x7+x9+x11)*voln*6.023e2
c  number of calcineurin molecules in cytoplasm
      cnum_Cn = (x4+x8+x10+x12)*volc*6.023e2
c  total number of calcineurin molecules
      total_Cn = (x4+x8+x10+x12)*volc*6.023e2 + 
     +     (x3+x7+x9+x11)*voln*6.023e2
      
    
c*********** NFAT simulations *******************
		          
		 
            fx1 = ( ak1*x5 - ak2*x1 + ak17*x2*vratio - ak18*x1 +
     +           ak15*x9 - ak16*x1*x3 )
            
            fx2 = ( ak1*x6 - ak2*x2 + ak18*x1*vratioc 
     +           -ak17*x2 + ak15*x10 - ak16*x2*x4 )
            
            eki = 20*(x13**2/(x13**2+1**2))*(x13**2/(x13**2+10**2))
            fx3 = ( -ak11*x5*x3 + ak12*x7 + ak5*x4*vratio - ak6*x3 + 
     +           ak15*x9 - ak16*x3*x1 + ak19*x11*eki - ak20*x3 )
c     +           ak15*x9 - ak16*x3*x1 + ak19*x11*x13**3 - ak20*x3 )
            
            eki = 20*(x14**2/(x14**2+1**2))*(x14**2/(x14**2+10**2))
            fx4 = ( -ak11*x6*x4 + ak12*x8 - ak5*x4 + ak6*x3*vratioc + 
     +           ak15*x10 - ak16*x4*x2 + ak19*x12*eki - ak20*x4 )
c     +           ak15*x10 - ak16*x4*x2 + ak19*x12*x14**3 - ak20*x4 )
            
            fx5 = ( -ak1*x5 + ak2*x1 - ak4*x5 + 
     +           ak3*x6*vratio - ak11*x5*x3 + ak12*x7 )
            
            fx6 = ( -ak1*x6 + ak2*x2 + ak4*x5*vratioc - ak3*x6 
     +           -ak11*x6*x4 + ak12*x8 )
            
            fx7 = ( ak11*x5*x3 - ak12*x7 + ak7*x8*vratio - ak8*x7 
     +           -ak13*x7 + ak14*x9 )
            
            fx8 = ( ak11*x6*x4 - ak12*x8 + ak8*x7*vratioc - ak7*x8 
     +           -ak13*x8 + ak14*x10 )
            
            fx9 = ( ak16*x1*x3 - ak15*x9 + ak13*x7 - ak14*x9 +
     +           ak9*x10*vratio - ak10*x9 )
            
            fx10 = ( ak16*x2*x4 - ak15*x10 + ak13*x8 - ak14*x10 + 
     +           ak10*x9*vratioc - ak9*x10 )
            
            eki = 20*(x13**2/(x13**2+1**2))*(x13**2/(x13**2+10**2))
c            fx11 = ( ak5*x12*vratio - ak6*x11 - ak19*x11*x13**3 + 
            fx11 = ( ak5*x12*vratio - ak6*x11 - ak19*x11*eki + 
     +           ak20*x3 )
            
            eki = 20*(x14**2/(x14**2+1**2))*(x14**2/(x14**2+10**2))
c            fx12 = (-ak5*x12 + ak6*x11*vratioc - ak19*x12*x14**3 + 
            fx12 = (-ak5*x12 + ak6*x11*vratioc - ak19*x12*eki + 
     +           ak20*x4)
            
c     fx13 = (ak21*x14*vratio - ak22*x13-ak19*x11*x13**3 + ak20*x3)
            
c     fx14 = (-ak21*x14+ak22*x13*vratioc - ak19*x12*x14**3 + ak20*x4)
          

cc------------------------------------------------------------------------            
c perform integration
            
            x1 = x1 + dt*fx1 
            x2 = x2 + dt*fx2 
            x3 = x3 + dt*fx3 
            x4 = x4 + dt*fx4 
            x5 = x5 + dt*fx5 
            x6 = x6 + dt*fx6 
            x7 = x7 + dt*fx7 
            x8 = x8 + dt*fx8 
            x9 = x9 + dt*fx9 
            x10 = x10 + dt*fx10 
            x11 = x11 + dt*fx11 
            x12 = x12 + dt*fx12            
    

c     number of NFAT molecules in nucleus
         anum_NFAT = (x1+x5+x7+x9)*voln*6.023e2
c     number of NFAT molecules in cytoplasm
         cnum_NFAT = (x2+x6+x8+x10)*volc*6.023e2
c     total number of NFAT molecules
         total_NFAT = (x2+x6+x8+x10)*volc*6.023e2 + 
     +        (x1+x5+x7+x9)*voln*6.023e2
         
c     fraction of NFAT in transcriptionally active form
         fra_NFAT = (x9*voln*6.023e2)/((x2+x6+x8+x10)*volc*6.023e2 +
     +        (x1+x5+x7+x9)*voln*6.023e2)
            
 
c  number of calcineurin molecules in nucleus
         anum_Cn = (x3+x7+x9+x11)*voln*6.023e2
c     number of calcineurin molecules in cytoplasm
         cnum_Cn = (x4+x8+x10+x12)*volc*6.023e2
c     total number of calcineurin molecules
         total_Cn = (x4+x8+x10+x12)*volc*6.023e2 + 
     +        (x3+x7+x9+x11)*voln*6.023e2
     
      vnfat(1) = x1       ! [NFAT]c (uM)
      vnfat(2) = x2     ! [NFAT]n (uM)
      vnfat(3) = x3     ! [C*]n (uM)
      vnfat(4) = x4     ! [C*]c (uM)
      vnfat(5) = x5    ! [NFAT:Pi]n (uM)
      vnfat(6) = x6      ! [NFAT:Pi]c (uM)
      vnfat(7) = x7      ! [NFAT:Pi:C*]n (uM)
      vnfat(8) = x8     ! [NFAT:Pi:C*]c (uM)
      vnfat(9) = x9     ! [NFAT:C*]n (uM)
      vnfat(10) = x10    ! [NFAT:C*]c (uM)
      vnfat(11) = x11    ! [C]n (uM)
      vnfat(12) = x12    ! [C]c (uM)

      return
      end 

NFκB subroutine

      subroutine nfkb(time,dt,vnfkb)
      implicit double precision (a-h,o-z)
      dimension vnfkb(19)

c------ T cell model--------------------------------------
c**By Pei-Chi Yang
c**Copyright 2007 Saleet Jafri Lab. All rights reserved.
c**9-10-2007 
c---------------------------------------------------------

cc--------------------------------
c********** NFkB *****************   
c define parameter values

      ak1 = 0.000614            ! rate  of association of NFkB and IkB:Pi 
      ak2 = 0.00184            ! rate of dissociation of NFkB:IkB:Pi 
      ak3 = 0.0020              ! rate of dephosphorylation of NFkB:IkB 
      ak4 = 0.0010               ! rate of phosphorylation of NFkB:IkB 
      ak5 = 0.00026              ! rate of translocation of (NFkB)n to(NFkB)c  
      ak6 = 0.0134               ! rate of translocation of (NFkB)c to(NFkB)n  
      ak7 = 0.010                ! rate of translocation of (IkB)n to(IkB)c 
      ak8 = 0.02                 ! rate of translocation of (IkB)c to(IkB)n  
      ak9 = 0.000034             ! rate  of translocation of(NFkB:IkB:Pi)n to(NFkB:IkB:Pi)c 
      ak10 = 0.000034            ! rate of translocation of(NFkB:IkB:Pi)c to(NFkB:IkB:Pi)n 
      ak11 = 0.02                ! rate  of translocation of(NFkB:IkB)n to (NFkB:IkB)c 
      ak12 = 0.000034            ! rate of translocation of(NFkB:IkB)c to (NFkB:IkB)n 
      ak13 = 0.00092            ! rate of translocation of (C*)n to (C*)c  
      ak14 = 0.0019             ! rate of translocation of (C*) to (C*)n 
      ak15 = 1.0                ! rate deactivation of Calcinurin 
      ak16 = 0.000000001        ! rate activation of Calcinurin 
      ak17 = 0.00092            ! rate  of translocation of (C)n to (C)c 
      ak18 = 0.0019             ! rate of translocation of (C)c to (C)n 
      ak19 = 0.5                ! rate of translocation of (Ca 2+)n to (Ca 2+)c
      ak20 = 0.21               ! rate of translocation of (Ca 2+)c to (Ca 2+)n 
      ak21 = 0.02               ! rate constant of degradation of (IkB:Pi)c  
      ak22 = 0.0022/12.4       ! rate of synthesis of new IkB
      ak23 = 0.000000036      ! rate activation of IKK by PKCtheta
      ak24 = 0.00008          ! rate deactivation of IKK by PKCtheta
      ak25 = 0.0000016        ! rate activation of IKK by PKC-alpha
      ak26 = 0.0008            ! rate deactivation of IKK by PKC-alph
      ak27 = 0.0016           ! rate activation of IKK by Cn 
      ak28 = 0.0006           ! rate deactivation of IKK by Cn
      ak29 = 0.009           ! rate of translocation of (Ikk)c to(Ikk)n 
      ak30 = 0.02             ! rate of translocation of (Ikk)n to(Ikk)c  
      tr2a = 0.00154
      tr2 = 0.0000165
      tr3 = 0.00028
	  
c import variable values from main
            x1 = vnfkb(1)
            x2 = vnfkb(2)
            x3 = vnfkb(3)
            x4 = vnfkb(4)
            x5 = vnfkb(5)
            x6 = vnfkb(6)
            x7 = vnfkb(7)
            x8 = vnfkb(8)
            x9 = vnfkb(9)
            x10 = vnfkb(10)
            x11 = vnfkb(11)
            x12 = vnfkb(12)
            x13 = vnfkb(13)
            x14 = vnfkb(14)
            x15 = vnfkb(15)
            x16 = vnfkb(16)
            x17 = vnfkb(17)
            x18 = vnfkb(18)
            x19 = vnfkb(19) 
   
c initial conditions
       if (time.lt.2*dt) then
c steady state values for ca = 100 nM
        x1 = 7.17792E+00       ! [NFkB]n (nM)
        x2 = 7.31124E-02       ! [NFkB]c (nM)
        x3 = 2.03563E-01       ! [IkB]n (nM)
        x4 = 5.25426E-02       ! [IkB]c (nM)
        x5 = 2.34342E-01       ! [NFkB:IkB:Pi]n (nM)

        x6 = 1.07661E-01       ! [NFkB:IkB:Pi]c (nM)
        x7 = 2.37664E-01       ! [NFkB:IkB]n (nM)
        x8 = 5.29613E+01       ! [NFkB:IkB]c (nM)
        x9 = 5.05386E-02 	! #[C*]n (nM)#
        x10 = 9.14784E-03 	! #[C*]c (nM)#

        x11 = 4.91984E+01      ! #[C]n (nM)#
        x12 = 9.71085E+00      ! #[C]c (nM)#
        x13 = vnfkb(13)		! Ca nuclear
        x14 = vnfkb(14)		! Ca cytosol
        x15 = 0.0              ! [IKK*:PKCheta] (%)
        x16 = 0.0              ! [IKK*:PKCalpha:Cn] (%)
        x17 = 0.0              ! [IKK*]c (%)
        x18 = 0.0              ! [IKK*]n (%)
        x19 = 0.0              ! IkBt
        endif
		 
	  PKCtheta = 2000.0         ! (nM)
      Ptotal = 1000.0           ! total concentration of PKC-alpha (nM)
                
      H_kB = 1.5                   ! Hill coefficient                                  
      ck1_kB = 22000.0              ! dissociation constant (nM)                           
      ck2_kB = 2.0                   ! stoichiometric nomalization constant                  
      dk1_kB = 10.2                ! dissociation constant (nM)                            
      dk2_kB = 1.0                   ! stoichiometric nomalization constant                    

      volc = 269                ! cytosolic volume (um^3)
      voln = 113                ! nuclear volume (um^3)
      vratio = volc/voln  
      vratioc = voln/volc 





      total_IkB = ((x3+x5+x7)*voln+(x4+x6+x8)*volc)/(volc+voln)
c total IkB concentration with respect to cell volume
      ctotal_IkB = ((x3+x5+x7)*voln+(x4+x6+x8)*volc)/(volc)
c total IkB concentration with respect to cytosolic volume
      total_NFkB = ((x1+x5+x7)*voln+(x2+x6+x8)*volc)/(volc+voln)
c total NFkB concentration with respect to cell volume
      ctotal_NFkB = ((x1+x5+x7)*voln+(x2+x6+x8)*volc)/(volc)
c total NFkB concentration with respect to cytosolic volume
      fra_NFkB = ( x1*vratioc/ctotal_NFkB )
c fraction activity of NFkB
    

cc------------------------------------------------------
c************* NFkB *****************             

            fx1  = ( -ak5*x1 + ak6*x2*vratio - ak1*x1*x3 + ak2*x5 )
            fx2  = ( -ak6*x2 + ak5*x1*vratioc - ak1*x2*x4 + ak2*x6 ) 
            fx3 = ( -ak7*x3 + ak8*x4*vratio - ak1*x1*x3 + ak2*x5 ) 
            fx4 = ( -ak8*x4 + ak7*x3*vratioc - ak1*x2*x4 + ak2*x6 
     +           - ak21*x4 + ak22*x19 ) 
            
            fx5 = (ak1*x1*x3 - ak2*x5 - ak3*x5 + ak4*x7*x18 +
     +           ak10*x6*vratio - ak9*x5 )
           
            fx6 = ( ak1*x2*x4 - ak3*x6 + ak4*x8*x17 - ak2*x6 + 
     +           ak9*x5*vratioc - ak10*x6 )    
                      
            fx7 = ( ak3*x5 + ak12*x8*vratio - ak11*x7 - ak4*x7*x18 )

            fx8 = ( ak3*x6 - ak4*x8*x17 +ak11*x7*vratioc - ak12*x8 )
                                   
            fx9 = ( -ak15*x9 + ak16*x13*x13*x13*x11 - ak13*x9 + 
     +           ak14*x10*vratio) 
            fx10 = ( -ak15*x10 + ak16*x14*x14*x14*x12 + 
     +           ak13*x9*vratioc - ak14*x10 )
            
            fx11 = ( ak15*x9 - ak16*x11*x13*x13*x13 + 
     +           ak18*x12*vratio - ak17*x11 )
            
            fx12 = ( ak15*x10 - ak16*x12*x14*x14*x14 - ak18*x12 +
     +           ak17*x11*vratioc )
            
c     fx13 = ( ak15*x9 - ak16*x11*x13*x13*x13 + ak20*x14*vratio - ak19*x13 )  
c     fx14 = ( ak15*x10 - ak16*x12*x14*x14*x14 - ak20*x14 + ak19*x13*vratioc )
               
            
            fx15 = ( ak23*(1-x15)*PKCtheta - ak24*x15 )
                                  
            fx16 = ( ak25*(1-x16)*PKC - ak26*x16 + 
     +           ak27*(1-x16)*x10 - ak28*x16 )

            fx17 = (fx15+fx16 + ak30*x18*vratioc - ak29*x17) 

            fx18 = ( ak29*x17*vratio - ak30*x18 )  

            fx19 = (tr2a + tr2*x1 - tr3*x19 )

cc---------------------------------------------------------------------

c perform integration
            
            x1 = x1 + dt*fx1 
            x2 = x2 + dt*fx2 
            x3 = x3 + dt*fx3 
            x4 = x4 + dt*fx4 
            x5 = x5 + dt*fx5 
            x6 = x6 + dt*fx6
            x7 = x7 + dt*fx7 
            x8 = x8 + dt*fx8 
            x9 = x9 + dt*fx9 
            x10 = x10 + dt*fx10 
            x11 = x11 + dt*fx11 
            x12 = x12 + dt*fx12  
            x15 = x15 + dt*fx15
            x16 = x16 + dt*fx16
            x17 = x17 + dt*fx17
            x18 = x18 + dt*fx18
            x19 = x19 + dt*fx19

   
		 total_IkB = ((x3+x5+x7)*voln+(x4+x6+x8)*volc)/(volc+voln)
c total IkB concentration with respect to cell volume
         ctotal_IkB = ((x3+x5+x7)*voln+(x4+x6+x8)*volc)/(volc)
c total IkB concentration with respect to cytosolic volume
         total_NFkB = ((x1+x5+x7)*voln+(x2+x6+x8)*volc)/(volc+voln)
c total NFkB concentration with respect to cell volume
         ctotal_NFkB = ((x1+x5+x7)*voln+(x2+x6+x8)*volc)/(volc)
c total NFkB concentration with respect to cytosolic volume
         fra_NFkB = ( x1*vratioc/ctotal_NFkB )
c fraction activity of NFk
         total_IkB = ((x3+x5+x7)*voln+(x4+x6+x8)*volc)/(volc+voln)

      vnfkb(1) = x1
      vnfkb(2) = x2
      vnfkb(3) = x3
      vnfkb(4) = x4
      vnfkb(5) = x5
      vnfkb(6) = x6
      vnfkb(7) = x7
      vnfkb(8) = x8
      vnfkb(9) = x9
      vnfkb(10) = x10
      vnfkb(11) = x11
      vnfkb(12) = x12
      vnfkb(15) = x15
      vnfkb(16) = x16
      vnfkb(17) = x17
      vnfkb(18) = x18
      vnfkb(19)  = x19
     
     
      return
      end 

JNK subroutine

      subroutine jnk(time,dt,vjnk)
      implicit double precision (a-h,o-z)
      dimension vjnk(23)

cc--------------------------------
c********** JNK/JUN  *****************   

    
      ak13 = 0.00092            ! rate of translocation of (C*)n to (C*)c  
      ak14 = 0.0019             ! rate of translocation of (C*) to (C*)n 
      ak15 = 1.0                ! rate deactivation of Calcinurin 
      ak16 = 0.000000001        ! rate activation of Calcinurin 
      ak17 = 0.00092            ! rate  of translocation of (C)n to (C)c 
      ak18 = 0.0019             ! rate of translocation of (C)c to (C)n 
      ak19 = 0.5                ! rate of translocation of (Ca 2+)n to (Ca 2+)c
      ak20 = 0.21               ! rate of translocation of (Ca 2+)c to (Ca 2+)  

c only JNK

      ak35 = 0.016              !k1 !k45
      ak36 = 0.00038        !k2 !k46
      ak41 = 0.00001        !k7 !k51
      ak42 = 0.00025        !k8 !k52
     
      ak37 = 0.00012            !k3 !k47
      ak38 = 0.0075         !k4 !k48
      ak39 = 0.00001       !k5 !k49
      ak40 = 0.00088        !k6 !k50

c Import variable values from main    
      x9 = vjnk(9)	! #[C*]n (nM)#
      x10 =vjnk(10)  	! #[C*]c (nM)#
      x11 = vjnk(11)       ! #[C]n (nM)#
      x12 = vjnk(12)     ! #[C]c (nM)#
      x13 = vjnk(13)		! [Ca]n (nM)
      x14 = vjnk(14)           ! [Ca]c (nM)
      x19 = vjnk(19)             ! [JNK*] (%)
      x20 = vjnk(20)              ! [c-Jun] (%)
      x21 = vjnk(21)              ! [JNK:Cn] (%)
      x22 = vjnk(22)              ! [JNK:PKC] (%)
      x23 = vjnk(23)            ! [JNK:PKC:Cn](%)


c Initial Conditions 

	if (time.lt.2*dt) then

c steady state values for ca = 100 nM
        
        x9 = 5.05386E-02	! #[C*]n (nM)#
        x10 = 9.14784E-03 	! #[C*]c (nM)#
        x11 = 4.91984E+01      ! #[C]n (nM)#
        x12 = 9.71085E+00     ! #[C]c (nM)#
        x13 = vjnk(13)		! [Ca]n (nM)
        x14 = vjnk(14)           ! [Ca]c (nM)
        x19 = 0.0              ! [JNK*] (%)
        x20 = 0.0              ! [c-Jun] (%)
        x21 = 0.0              ! [JNK:Cn] (%)
        x22 = 0.0              ! [JNK:PKC] (%)
  
        endif

        PKCtheta = 2000.0         ! (nM)
        Ptotal = 1000.0           ! total concentration of PKC-alpha (nM)

        IKKtotal = 10000.0        ! (nM)

        JNKtotal = 1000.0         ! (nM)
        cJuntotal = 20.0          ! (nM)

        cFos = 50.0               ! (nM)
        AP1 = 50.0                ! (nM)
        H_jk = 1.5                   ! Hill coefficient
        ck1_jk = 2200.0              ! dissociation constant (nM)
        ck2_jk = 2.0                 ! stoichiometric nomalization constant
        dk1_jk = 10.2                ! dissociation constant (nM)
        dk2_jk = 1.0                 ! stoichiometric nomalization constant

        volc = 269                ! cytosolic volume (um^3)
        voln = 113                ! nuclear volume (um^3)
        vratio = volc/voln
        vratioc = voln/volc




cc------------------------------------------------------
c************* JNK *****************             
c Define derivatives (RHS equations)
            
           
            fx9 = ( -ak15*x9 + ak16*x13*x13*x13*x11 - ak13*x9 +
     +           ak14*x10*vratio)
            fx10 = ( -ak15*x10 + ak16*x14*x14*x14*x12 +
     +           ak13*x9*vratioc - ak14*x10 )


            fx11 = ( ak15*x9 - ak16*x11*x13*x13*x13 +
     +           ak18*x12*vratio - ak17*x11 )

            fx12 = ( ak15*x10 - ak16*x12*x14*x14*x14 - ak18*x12 +
     +           ak17*x11*vratioc )
  
            fx19 = 0.0
            fx20 = 0.0

c only JNK
            fx21 = ( ak35*(1-x21-x22-x23)*x10 - ak36*x21 +
     +           ak42*x23 - ak41*x21*PKCtheta )

            fx22 = ( ak37*(1-x21-x22-x23)*PKCtheta - ak38*x22 +
     +           ak40*x23 - ak39*x22*x10 )

            fx23 = ( ak39*x22*x10 - ak40*x23 +
     +           ak41*x21*PKCtheta - ak42*x23 )
cc---------------------------------------------------------------------

c perform integration
            

            x9 = x9 + dt*fx9
            x10 = x10 + dt*fx10
            x11 = x11 + dt*fx11
            x12 = x12 + dt*fx12
            x19 = x19 + dt*fx19
            x20 = x20 + dt*fx20
            x21 = x21 + dt*fx21
            x22 = x22 + dt*fx22
            x23 = x23 + dt*fx23

            vjnk(9) = x9
            vjnk(10) = x10
            vjnk(11) = x11
            vjnk(12) = x12
            vjnk(13) = x13
            vjnk(14) = x14
            vjnk(19) = x19
            vjnk(20) = x20
            vjnk(21) = x21
            vjnk(22) = x22
            vjnk(23) = x23
           
      return
      end 


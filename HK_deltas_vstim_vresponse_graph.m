function HK_deltas_vstim_vresponse_graph()
    max = 0.5;
    min = 0.5;
    for counter = min:max
     ggapval = counter*1;
     disp("ggapval="+ggapval);
     A=simulate_process(ggapval);
     x = A(:,1);
	 %y = A(:,100:115);
     Ng=200;
	 %Original 100:1:105
	 range = 99:1:135;
	 %y = A(:,range);
	 y = A(:,range);
	 
	 save('y.txt','y','-ascii');
	 
     
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     f = figure('visible','on');
     plot(x,y,'linewidth', 3); title("G gap = "+ggapval);
     fileName = strcat('Image',num2str(ggapval),'.png');
     saveas(gcf,fileName);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
     %exportgraphics(f,"C:/Users/iqbal/Downloads/Capillary-Kir-Model-master/Capillary-Kir-Model-master/scripts/Ours/test.pdf","Append",true);
     %exportgraphics(f,"test.pdf","Append",true)
     
    end
end



function returnedVal = simulate_process(g_gap_value)

	tic
	%TO FIX --> display each variable one by one to see what is unstable 
	%Setting up Variables 
	dt = 0.001;%0.001;
	loop = 600000;%1200000;%2500000;%600000;%ceil(1000/dt);  %30000
	Ng= 200; % 1000
	 
	r1 =  0;%-0.75 + 1.5*rand;
	Vm = (-33+r1)*ones(1,Ng);
	%Vm(1) = -140;   %mV
	cm = 9.4; %micro farad / cm^2 %specific capacitane



	 

	a = 0.01;%0.0238; %axon radius unit is milimiter = 10 micrometers
	rho = 0.0354/(0.36^2); %??? %resistivity <--- NOT USED
	dx = 0.06; %cm cell size 60 micrometer
	%g_gap = 9.4*200*0.00009*9/25 % # of channels * conductance in nano Siemens
	%g_gap = 0.05;%33.999;%33.95;%0.94;%33.5;%150*700*0.00009; ??? 34 cuttoff
	
	g_gap = g_gap_value;%33.999;%33.95;%0.94;%33.5;%150*700*0.00009; ??? 34 cuttoff
	

	%Rho = 35.4 mV uA^-1 cm

	eki1 = (g_gap*dt)/(dx^(2)*cm);
	eki2 = dt/cm;
	t=0;
	F = 9.6485d4;%      F (Faraday)  [C/mol]
	R = 8.314d3;%       R (universal gas) [mJ mol-1 K-1]

	 

	size(Vm);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Stimulation protocol
	current_stim = false;     % current stimulus

	r = 0;%-10 + 20*rand;
	K_i = 150+r;%140 maybe?                       % [mM] intracellular potassium concentration
	 
	K_out=3*ones(1,loop);%5*ones(1,loop);%3<---set true
	 


	%% Current stimulation
	current = -1;        % [pA]   Injected current

	%% model parameters 

	%% constant parameters
	R  = 8314.0;	% [mJmol-1K-1]	gas constant
	T = 293; % [K] absolute temperature
	F  = 96485.0;	% [Cmol-1] Faraday's constant
	RT_F = R*T/F;   % RT/F
	z_K  = 1;       % K ion valence

	%% Kir channel characteristic
	delta_V_kir  = 25; % [mV]	voltage diff at half-max.
	G_kirbar = 0.94;%0.002*220;%3.93*0.18;% [nS/mM^0.5] inward rectifier constant
	n_kir = 0.5;	% inward rectifier constant
	k_kir = 7;	% [mV]	inward rectifier slope factor

	%% Background current
	r2 = 0;%-0.75 + 1.5*rand;
	E_bg = -30+r2;  % [mV] resting membrane potential
	ratio=0.7; %0.3 later
	G_bg = ratio*G_kirbar;  % [nS] background conductance

	I_bg = zeros(1, Ng)
    I_kir = zeros(1, Ng)
    distance_m = zeros(1, Ng)
    vstims = zeros(1, Ng)
    vresps = zeros(1, Ng)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%loop 300000
	%Ng 1000
	%Running experiment
	A=zeros(loop,Ng+1);
	I_app = zeros(1,Ng);
	%I_app = zeros(1,loop);
	for j = 1:loop 
		t =(j-1)*dt+dt;
	   % time(j) = time(j-1) + dt;
	   
			
	   
			K_o = 5;
			   if (t>=100) && (t<= 400)
					I_app(100) = 50.0;%10*9.4*0.5;%-1;               %uA/cm^2, externally applied current 
					%I_app(1:10) = 50.0;%10*9.4*0.5;%-1;               %uA/cm^2, externally applied current 
			   else
			        I_app(100) = 0.0;
			%disp(t)
			%disp(I_app(1:100
			
			   end
				
			K_o = K_out(j); %K_out(t);
		 
	     
		 for kk = 1:Ng % 500 axonal segments 
			
			
			
			%% Potassium reversal potential
			E_K = RT_F/z_K*log(K_o./K_i);     %[mV]
			
			%% Membrane currents
			I_bg(kk) = G_bg.*(Vm(kk) - E_bg);        %[pA] lumped background current
			I_kir(kk) = G_kirbar.*(K_o).^n_kir .*((Vm(kk) - E_K)./(1 + exp((Vm(kk) - E_K - delta_V_kir)./k_kir)));   %[pA] whole cell kir current
			
			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			random_number = 1;%0.5*rand + 0.75;
				   
			%Calculation of final membrane potential, dependent on time and voltage
		   if kk==1 
		   Vm(kk) = Vm(kk) + random_number*eki1*(Vm(kk+1)-Vm(kk))-eki2*(I_bg(kk)+  ...
			   I_kir(kk)+I_app(kk));
		   elseif kk==Ng
		   Vm(kk) = Vm(kk) + random_number*eki1*(Vm(kk-1)-Vm(kk))-eki2*(I_bg(kk)+ ...
				   I_kir(kk)+I_app(kk));
		   %elseif kk >= 99 || kk <=101
		   %Vm(kk) = Vm(kk) + random_number*(eki1)*(Vm(kk+1)+Vm(kk-1)-2*Vm(kk))-eki2* ...
		   %   (I_bg(kk)+I_kir(kk)+I_app(kk));
		   elseif kk==99
		   Vm(kk) = Vm(kk) + random_number*(eki1)*(0.6*(Vm(kk+1)-Vm(kk))+Vm(kk-1)-Vm(kk))-eki2* ...
		      (I_bg(kk)+I_kir(kk)+I_app(kk));
		   elseif kk==100 ! cell with injected current
		   Vm(kk) = Vm(kk) + random_number*(eki1)*0.6*(Vm(kk+1)+Vm(kk-1)-2*Vm(kk))-eki2* ...
		      (I_bg(kk)+I_kir(kk)+I_app(kk));  
		   elseif kk==101
		   Vm(kk) = Vm(kk) + random_number*(eki1)*(Vm(kk+1)-Vm(kk)+0.6(Vm(kk-1)-Vm(kk)))-eki2* ...
		      (I_bg(kk)+I_kir(kk)+I_app(kk));			 
		   else          
		   Vm(kk) = Vm(kk) + random_number*eki1*(Vm(kk+1)+Vm(kk-1)-2*Vm(kk))-eki2* ...
			   (I_bg(kk)+I_kir(kk)+I_app(kk));
		   end
		   
		    
			distance_m(kk) = kk*dx;
			
		    if kk == 100
				vstims(kk) = Vm(kk);
			else
			    vresps(kk) = Vm(kk);
		    end

		%disp(kk)  
		 end
		 
	   A(j,1)=t;
	   A(j,2:Ng+1)=Vm; %%Vm(1:Ng);
	 %disp(Vm);
	%plot(t,Vm)
	end
    
	%Delta Stim calculation
	%99:1:135
	%1:1:200
	At399=A(399999,99:1:135);
	save('At399.txt','At399','-ascii')
	
	At99=A(99999,99:1:135);
	save('At99.txt','At99','-ascii')
    B1 = abs(A(399999,99:1:135) - A(99999,99:1:135));
	
	save('B1.txt','B1','-ascii')
	
    %Delta Resp calculation
	%B2 = abs(A(100000,99:1:135));
	B2 = abs(A(99999,99:1:135));
	
	save('B2.txt','B2','-ascii')
	
    %Delta Resp divided by Delta Stim calculation
    %D = C1(:)/B2;
	disp(">>>>>>>>>>>>>>>>>>>>>"+B1(1))
    %D = B1(1)./B1;
	D = B1./B1(1)
	size(D)
	save('D.txt','D','-ascii');
    
    %To compute the distance that we need in the plot
    %I'm using dx then approximate the dx*Ng = ceil(0.6*200) = 12
    dx
    Ng
    dx*Ng
    endval = ceil(dx*Ng)
    %I created an array of graduat distances using matlab notation
    %Starts from 0, with 0.6 steps until the endvalue 12=0.6*200
    distance_m  = 99:1:135;
    distance_m  = dx*(99:1:135);
	save('distance_m.txt','distance_m','-ascii');
    size(distance_m)
    size(D)
    %plot Ratios to distance using dots
    plot(distance_m,D, '.', 'markersize', 8)
	c = polyfit(distance_m,D,1);
	

	disp(['Equation is y = ' num2str(c(1)) '*x + ' num2str(c(2))])
	y_est = polyval(c,distance_m);

	hold on
	%plot(distance_m,y_est,'r--','LineWidth',2)
	plot(distance_m,y_est,'LineWidth',2)
	hold off


	%save('ratios.txt','D','-ascii');
	%save('abs_delta_t399mint99.txt','C1','-ascii');
    %save('abs_delta_t390mint101.txt','C2','-ascii');
	%save('Vm_vs_T_t99_t399.txt','B1','-ascii');
    %save('Vm_vs_T_t101_t390.txt','B2','-ascii');
	%save('Vm_vs_T.txt','A','-ascii');

	x = A(:,1);
	%y = A(:,100:115);
	y = A(:,100:Ng);
	
    returnedVal = A;
	
   
    
	toc
 
end









WILSON-COWAN: DISTANZA DALL'EQUILIBRIO  (versione di ottobre 2026)
=================================================================

Modello di Wilson-Cowan stocastico sbilanciato (popolazioni eccitatorie e
inibitorie, chiE = 70%, chiI = 30%) su una griglia di parametri di
disaccoppiamento sinaptico delta_EI x delta_IE. Si studiano il tasso di
produzione di entropia W, la violazione della relazione di
fluttuazione-dissipazione (FDT) e la temperatura efficace beta_eff.

Riferimento principale:

  M. K. Nandi, A. de Candia, A. Sarracino, H. J. Herrmann,
  L. de Arcangelis, "Fluctuation-dissipation relations in the
  imbalanced Wilson-Cowan model", Phys. Rev. E 107, 064307 (2023).
  DOI: 10.1103/PhysRevE.107.064307


CONVENZIONI COMUNI AI NOTEBOOK DI OTTOBRE
-----------------------------------------

  - Solo h = 1e-6 (il valore dell'articolo): file
    grid_dynamics_7030_h1e-06_sigmaFIX.npz.

  - I dati vengono arrotondati a 9 cifre decimali al caricamento: i
    valori sotto 1e-9 (per esempio E0 ~ 1e-12 dove dovrebbe essere 0)
    sono rumore del solutore e vengono portati a zero.

  - La regione proibita (w_EI < 0 o w_IE < 0, cioe' le righe
    delta_EI = -7.0 e -6.9) e' esclusa da tutte le analisi.

  - Nel primo quadrante (E0 = 0, A_EI = 0) il sistema e' 1D e
    all'equilibrio: W = 0/0. Questi punti sono esclusi dalle analisi
    di beta_eff e della FDT.

  - Nessuna quantita' salvata nei .npz viene ricalcolata: i notebook
    calcolano solo quantita' derivate.


1. simulation_7030.py
=====================

Simulazione del modello sulla griglia delta_EI x delta_IE. Per ogni
punto:

  - integra numericamente (Numba, parallelizzato con prange) le
    equazioni di Langevin non lineari per il numero di neuroni attivi
    eccitatori (k) e inibitori (l), con dt = 1e-3 ms e T = 32000 ms
    (la seconda meta' usata per le statistiche stazionarie);

  - calcola le medie stazionarie E0, I0 e le fluttuazioni xiE, xiI;

  - calcola analiticamente (linearizzazione attorno al punto fisso,
    Eq. (A16)-(A21) e (A32)-(A34) di Nandi et al.) la matrice di drift
    A_EE, A_EI, A_IE, A_II, gli autovalori lambda1, lambda2, lambda_re,
    lambda_im, la matrice nelle variabili (Sigma, Delta) A_x, A_y, A_z,
    A_w, la covarianza stazionaria sigma11, sigma12, sigma22 e Sigma0,
    Delta0 al punto fisso.

I risultati per ogni valore di h vengono salvati in
grid_dynamics_7030_h{h}.npz e .mat.

NOTA: la formula di sigma12 nello script contiene un errore di segno
(vedi correzione.py). I notebook usano i file corretti _sigmaFIX.npz.


2. correzione.py
================

Ricalcola sigma11, sigma12, sigma22 dai .npz prodotti da
simulation_7030.py, correggendo il segno del termine con H nella
formula di sigma12 (Eq. A33 di Nandi et al.):

  - formula errata:   sigma12 = -(G*(z*w + x*y) + 2*H*x*w) / denom
  - formula corretta: sigma12 = -(G*(z*w + x*y) - 2*H*x*w) / denom

e salva una copia con suffisso _sigmaFIX.npz, lasciando invariati gli
altri campi. In alternativa basta cambiare il segno del termine
2.0 * H * x * w da + a - direttamente in simulation_7030.py.


3. Wilson_Cowan_ottobre_senza_beta.ipynb
========================================
Il modello e il tasso di produzione di entropia W

CONTENUTO, IN ORDINE

  1) E0, I0, Sigma0, Delta0 in scala lineare e logaritmica (piu'
     log10(-Delta0) dove Delta0 < 0).

  2) Correlazioni e risposte (C e R in base Sigma-Delta, caso reale e
     complesso) in un punto scelto manualmente, con l'opzione di
     normalizzare C(t)/C(0) come nelle figure dell'articolo.

  3) Diagramma degli autovalori (ricostruzione Fig. 2 di Nandi et al.).

  4) Tasso di produzione di entropia

       W = -delta^2 / [(A_EE + A_II) E0 I0],
       delta = sqrt(chiE/chiI) A_EI I0 - sqrt(chiI/chiE) A_IE E0

     (formula dalle note di Dario, "Linearized Wilson-Cowan"):
     classificazione dei punti (W finito, W = 0/0, un punto con
     W = inf), mappe in scala log, W lungo linee a delta_EI e delta_IE
     fissati, numeratore e denominatore, diagnostica del collasso a 1D.

  5) Fluttuazione relativa sqrt(sigma11)/Sigma0.

  6) Diagnostica del punto con W = inf: e' un artefatto numerico sul
     bordo del primo quadrante (E0 ~ 1e-12 arrotondato a zero).


4. beta_ottobre.ipynb
=====================
La temperatura efficace beta_eff

beta_eff = chi(inf) / C_SigmaSigma(0), con chi(t) la risposta integrata
di Sigma. Poiche' tutti gli autovalori hanno parte reale negativa, ha
la forma chiusa

  beta_eff = -w / (det A * sigma11)

equivalente alle formule con gli autovalori (caso reale e complesso).

CONTENUTO, IN ORDINE

  1) Mappe di beta_eff > 0 e beta_eff < 0 (l'altro segno in nero), su
     scala fissa [-20, 20] e in scala logaritmica.

  2) Diagnostica del ramo complesso: i singoli pezzi della formula.

  3) Fluttuazione relativa sqrt(sigma11)/Sigma0.

  4) Relazione tra beta_eff e W (DA RIVEDERE): formula di beta_eff in
     funzione di Lambda = alpha*delta, valore all'equilibrio beta0 ed
     eccesso beta_eff - beta0, correlazioni di Spearman con W su tutto
     il piano e lungo 12 linee, regime lineare
     |beta_eff - beta0| ~ sqrt(W). Questa sezione va rieseguita.


5. FDR_ottobre_2.ipynb
======================
Violazione della relazione di fluttuazione-dissipazione

Con x(t) = C(t)/C(0) e y(t) = chi(t)/C(0) la FDT prevede
y(t)/y_inf = 1 - x(t). La violazione e' misurata da

  I^(inf) = integrale da 0 a infinito di |y(t)/y_inf - (1 - x(t))| dt

calcolato lungo 6 linee a delta_EI fissato e 6 a delta_IE fissato
(-5, -2.5, -0.5, 0, 2.5, 5).

SCELTE NUMERICHE

  - Griglia temporale adattata a ogni punto, costruita sui suoi
    autovalori: passo = 1/50 del tempo caratteristico piu' rapido
    (decadimento o periodo di oscillazione), fine a 21 volte il tempo
    di rilassamento piu' lento (e^-21 < 1e-9, la precisione dei dati).

  - y_inf calcolato come limite t -> inf della stessa formula usata
    per chi(t), cosi' che l'integrando vada esattamente a zero.

CONTENUTO

Controlli sulla griglia temporale e sulle due formule di y_inf;
I^(inf) lungo le linee; confronto con W lungo le linee e su tutto il
piano (correlazioni di Spearman); diagnostica dei picchi, che cadono
sugli zeri di w (I^(inf) ~ 1/|w|, effetto della normalizzazione per
y_inf). In appendice il confronto tra le tre definizioni possibili:

  I^(inf),   I_T = I^(inf) / T,   I_tilde = I^(inf) / tau_slow

QUESTIONE APERTA: per ora si usa I^(inf), ma non e' ancora deciso quale
delle tre definizioni sia la piu' adatta.


ALTRI FILE
==========

grid_dynamics_7030_h1e-0X_sigmaFIX.npz
    Dati corretti per h = 1e-5, ..., 1e-9. I notebook di ottobre usano
    solo h = 1e-6.

Wilson_Cowan.ipynb, beta_eff.ipynb, FDR_integrale.ipynb
    Versioni precedenti dei notebook 3, 4 e 5, con il confronto tra
    h = 1e-6, 1e-7, 1e-8, 1e-9. FDR_integrale.ipynb contiene anche le
    metriche di Monti et al. (Phys. Rev. Research 7, 013301 (2025)),
    non incluse in FDR_ottobre_2.ipynb.

LICENSE
    Licenza del repository.
